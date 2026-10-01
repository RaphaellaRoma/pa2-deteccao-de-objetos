"""Inferência torchvision com NMS próprio em ambas as etapas do Faster R-CNN."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from PIL import Image
from geometry import nms
from mot_data import validate_detections


def own_nms_tensor(boxes, scores, iou_threshold):
    import torch
    xyxy = boxes.detach().cpu().numpy()
    xywh = xyxy.copy()
    xywh[:, 2:] -= xywh[:, :2]
    keep = nms(xywh, scores.detach().cpu().numpy(), iou_threshold)
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def own_batched_nms(boxes, scores, groups, iou_threshold):
    """Grupos são níveis FPN no RPN e classes nas caixas finais."""
    import torch
    xywh = boxes.detach().cpu().numpy().copy()
    xywh[:, 2:] -= xywh[:, :2]
    confidence = scores.detach().cpu().numpy()
    categories = groups.detach().cpu().numpy()
    selected = []
    for group in np.unique(categories):
        indices = np.flatnonzero(categories == group)
        selected.extend(indices[nms(xywh[indices], confidence[indices], iou_threshold)].tolist())
    # Desempate estável pelo índice original, inclusive entre categorias.
    selected = np.array(sorted(selected), dtype=int)
    selected = selected[np.argsort(-confidence[selected], kind='stable')]
    return torch.tensor(selected, dtype=torch.int64, device=boxes.device)


@contextmanager
def own_nms_context():
    """Escopo sequencial; restaura inclusive após exceção. Não usar em threads."""
    import torchvision.ops as ops
    import torchvision.ops.boxes as box_ops
    targets = [(ops, 'nms', own_nms_tensor), (ops, 'batched_nms', own_batched_nms),
               (box_ops, 'nms', own_nms_tensor), (box_ops, 'batched_nms', own_batched_nms)]
    previous = [(module, name, getattr(module, name)) for module, name, _ in targets]
    counts = {'nms': 0, 'batched_nms': 0}
    def wrap(name, implementation):
        def call(*args, **kwargs):
            counts[name] += 1
            return implementation(*args, **kwargs)
        return call
    try:
        for module, name, implementation in targets:
            setattr(module, name, wrap(name, implementation))
        yield counts
    finally:
        for module, name, implementation in previous:
            setattr(module, name, implementation)


@contextmanager
def cache_lock(directory):
    """Lock liberado pelo SO após interrupção; evita duas inferências no mesmo cache."""
    import os
    path = Path(directory)/'.writer.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Cache em uso por outra inferência: {directory}') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def choose_device(requested='auto'):
    import torch
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA solicitada, mas indisponível. Instale PyTorch CUDA ou use CPU.')
    if requested not in ('cpu', 'cuda'):
        raise ValueError('device deve ser auto, cpu ou cuda')
    return requested


def build_detector(config, device=None):
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    device = choose_device(device or config['device'])
    if config['torchvision_model'] != 'fasterrcnn_resnet50_fpn' or config['torchvision_weights'] != 'COCO_V1':
        raise ValueError('Modelo/configuração não suportados; não há troca automática de arquitetura')
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    ensure_weights(weights)
    model = fasterrcnn_resnet50_fpn(weights=weights,
                                  box_score_thresh=config['cache_score_threshold']).eval().to(device)
    return model, weights.transforms(), device, {'torch': torch.__version__, 'torchvision': torchvision.__version__}


def ensure_weights(weights):
    """IPv4 evita espera de IPv6 em redes locais; verifica hash publicado no nome."""
    import torch
    from urllib.parse import urlparse
    filename = Path(urlparse(weights.url).path).name
    expected = filename.rsplit('-', 1)[1].split('.')[0]
    target = Path(torch.hub.get_dir())/'checkpoints'/filename
    if target.exists():
        if not file_hash(target).startswith(expected):
            raise ValueError('Checkpoint do detector corrompido no cache do PyTorch')
        return target
    curl = shutil.which('curl.exe') or shutil.which('curl')
    if not curl:
        torch.hub.load_state_dict_from_url(weights.url, check_hash=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.pth.part')
    subprocess.run([curl, '-4', '-f', '-L', '--retry', '2', '--connect-timeout', '20',
                    '-C', '-', '-o', str(temporary), weights.url], check=True)
    if not file_hash(temporary).startswith(expected):
        raise ValueError('Download dos pesos falhou na verificação SHA256')
    temporary.replace(target)
    return target


def infer_frame(model, transform, path, frame, device):
    import torch
    with Image.open(path) as image:
        tensor = transform(image.convert('RGB')).to(device)
    with torch.inference_mode():
        prediction = model([tensor])[0]
    mask = prediction['labels'] == 1
    boxes = prediction['boxes'][mask].detach().cpu().numpy().copy()
    boxes[:, 2:] -= boxes[:, :2]
    scores = prediction['scores'][mask].detach().cpu().numpy()
    rows = np.column_stack([np.full(len(boxes), frame), boxes, scores])
    return validate_detections(rows)


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def code_hash(path):
    """Mesmo código em LF/CRLF mantém cache portátil entre checkouts."""
    return hashlib.sha256(Path(path).read_text(encoding='utf-8').encode('utf-8')).hexdigest()


def _write_json(path, value):
    path = Path(path)
    partial = path.with_suffix('.json.tmp')
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    partial.replace(path)


def cache_signature(sequence, config, versions):
    geometry = Path(__file__).with_name('geometry.py')
    return {'sequence': sequence.name, 'length': sequence.length,
            'image_size': [sequence.width, sequence.height],
            'model': config['torchvision_model'], 'weights': config['torchvision_weights'],
            'cache_score_threshold': config['cache_score_threshold'],
            # Backend +cpu/+cu não altera o contrato e permite retomada em GPU.
            'versions': {k: v.split('+')[0] for k, v in versions.items()}, 'nms': 'own-numpy-v1',
            'geometry_sha256': code_hash(geometry), 'adapter_sha256': code_hash(__file__)}


def _sync(device):
    if device == 'cuda':
        import torch
        torch.cuda.synchronize()


def benchmark(sequence, model, transform, device, output):
    """Cinco warmups + até 30 quadros completos; nenhuma subamostragem temporal."""
    frames = list(range(1, min(sequence.length, 35)+1))
    durations = []
    import torch
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with own_nms_context() as counts:
        for i, frame in enumerate(frames):
            _sync(device)
            start = time.perf_counter()
            infer_frame(model, transform, sequence.image_path(frame), frame, device)
            _sync(device)
            if i >= 5:
                durations.append(time.perf_counter()-start)
    mean = float(np.mean(durations)) if durations else None
    result = {'device': device, 'warmup_frames': min(5, len(frames)),
              'measured_frames': len(durations), 'mean_seconds_per_frame': mean,
              'estimated_validation_minutes': mean*1875/60 if mean is not None else None,
              'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1024**2 if device == 'cuda' else None,
              'peak_process_memory_mb': peak_process_memory_mb(),
              'nms_calls': counts, 'includes_image_io_and_own_nms': True}
    _write_json(output, result)
    return result


def peak_process_memory_mb():
    """Pico do processo (não só tensors); Windows e Unix."""
    import os
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        api = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        api.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD]
        if not api(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
            raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo')
        return counter.PeakWorkingSetSize/1024**2
    import resource
    import sys
    scale = 1 if sys.platform == 'darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale/1024**2


def extract_detections(sequence, config, cache_dir, model, transform, device, versions, max_frames=None):
    """Retomada por quadro e hash. Manifesto commitado somente após salvar cada NPY."""
    directory = Path(cache_dir)/sequence.name
    with cache_lock(directory):
        return _extract_locked(sequence, config, cache_dir, model, transform, device, versions, max_frames)


def _extract_locked(sequence, config, cache_dir, model, transform, device, versions, max_frames=None):
    directory = Path(cache_dir)/sequence.name
    directory.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(sequence, config, versions)
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['signature'] != signature:
            raise ValueError(f'Cache incompatível em {directory}; escolha outro --cache-dir')
    else:
        manifest = {'signature': signature, 'frames': {}, 'device_used': device, 'runtime_versions': versions}
    limit = min(sequence.length, max_frames or sequence.length)
    with own_nms_context():
        for frame in range(1, limit+1):
            path = sequence.image_path(frame)
            if not path.exists():
                raise FileNotFoundError(f'Imagem ausente: {path}. Baixe --package frames.')
            image_hash = file_hash(path)
            target = directory/f'{frame:06d}.npy'
            info = manifest['frames'].get(str(frame))
            if info and info['image_sha256'] == image_hash and target.exists() and info['detection_sha256'] == file_hash(target):
                continue
            rows = infer_frame(model, transform, path, frame, device)
            temporary = target.with_suffix('.npy.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, rows, allow_pickle=False)
            temporary.replace(target)
            manifest['frames'][str(frame)] = {'image_sha256': image_hash,
                                              'detection_sha256': file_hash(target), 'device': device,
                                              'runtime_versions': versions}
            manifest['device_used'] = device
            manifest['runtime_versions'] = versions
            _write_json(manifest_path, manifest)
            if frame == 1 or frame % 25 == 0 or frame == limit:
                print(f'{sequence.name}: detector {frame}/{limit}', flush=True)
    return load_cached_detections(sequence, config, cache_dir, max_frames=max_frames)


def load_cached_detections(sequence, config, cache_dir, max_frames=None):
    directory = Path(cache_dir)/sequence.name
    manifest_path = directory/'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Cache ausente para {sequence.name}; execute parte1.py detect')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    versions = manifest['signature']['versions']
    if manifest['signature'] != cache_signature(sequence, config, versions):
        raise ValueError('Modelo/configuração/código mudou: cache inválido')
    limit = min(sequence.length, max_frames or sequence.length)
    rows = []
    for frame in range(1, limit+1):
        path = directory/f'{frame:06d}.npy'
        info = manifest['frames'].get(str(frame))
        if not info or not path.exists() or file_hash(path) != info['detection_sha256']:
            raise ValueError(f'Cache incompleto ou corrompido: {sequence.name}, quadro {frame}')
        # Permite avaliar em outra máquina sem imagens; se presentes, confere identidade.
        image_path = sequence.image_path(frame)
        if image_path.exists() and file_hash(image_path) != info['image_sha256']:
            raise ValueError(f'Imagem mudou: {sequence.name}, quadro {frame}')
        values = validate_detections(np.load(path, allow_pickle=False), sequence.length)
        if np.any(values[:, 0] != frame):
            raise ValueError('Cache contém quadro incorreto')
        rows.append(values)
    return np.concatenate(rows) if rows else np.empty((0, 6))
