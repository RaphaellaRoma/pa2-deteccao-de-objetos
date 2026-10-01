"""Parte 1: detecções congeladas -> rastreador -> avaliação própria -> artefatos."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
from baseline import IoUTracker
from detection_metrics import evaluate_detection
from metrics import evaluate
from mot_data import (Sequence, load_config, valid_gt, evaluation_mask,
                      save_predictions, validate_detections)


def run_tracker(detections, length, config, tracker=None):
    """Entrada sem GT. Um update em cada quadro, inclusive nos vazios."""
    tracker = tracker or IoUTracker(config['association_iou'], config['max_age'])
    detections = validate_detections(detections, length)
    rows = []
    for frame in range(1, length+1):
        boxes = detections[detections[:, 0] == frame, 1:5]
        rows.extend(tracker.update(frame, boxes))
    return np.asarray(rows, float).reshape(-1, 6)


def evaluate_sequence(sequence, detections, config, source='public', max_frames=None):
    """Fonte substituível para estresse; mesmas caixas para AP e rastreamento."""
    length = min(sequence.length, max_frames or sequence.length)
    raw = sequence.ground_truth()
    raw = raw[raw[:, 0] <= length]
    gt = valid_gt(raw)
    det = validate_detections(detections, sequence.length)
    det = det[(det[:, 0] <= length) & (det[:, 5] >= config['score_threshold'])]
    # Não filtrar por GT antes da associação: isso vazaria respostas ao tracker.
    pred = run_tracker(det, length, config)
    det_mask = evaluation_mask(raw, det[:, 0], det[:, 1:5], config['ignore_iou'])
    pred_mask = evaluation_mask(raw, pred[:, 0], pred[:, 2:6], config['ignore_iou'])
    tracking = evaluate(gt, pred[pred_mask], config['evaluation_iou'])
    detection = evaluate_detection(gt, det[det_mask])
    count = tracking['gt_identities']
    result = {'sequence': sequence.name, 'source': source,
              'split': 'validation' if sequence.name in config['split']['validation'] else 'train',
              'frames': length, 'full_sequence': length == sequence.length,
              'density': len(gt)/length, 'score_threshold': config['score_threshold'],
              **tracking, **detection,
              'identity_ratio': tracking['predicted_identities']/count if count else None,
              'switches_per_gt_identity': tracking['id_switches']/count if count else None,
              'raw_predicted_identities': int(len(np.unique(pred[:, 1]))),
              'ignored_detections': int((~det_mask).sum()), 'ignored_predictions': int((~pred_mask).sum())}
    return result, pred


def evaluate_all(root, config, output, sources=('public',), split='all', cache_dir='data/detections/torchvision',
                 max_frames=None, detections_dir=None):
    """Salva resultados e MOT bruto. Comparação completa exige ambas as fontes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = config['split']['train']+config['split']['validation'] if split == 'all' else config['split'][split]
    results, inputs = [], {}
    for name in names:
        sequence = Sequence.load(root, name, config['public_detector'])
        for source in sources:
            # Detector torchvision é obrigatório nas três sequências de validação.
            if source == 'torchvision' and split == 'all' and name not in config['split']['validation']:
                continue
            if source == 'public':
                detections = sequence.public_detections()
                input_path = sequence.directory/'det/det.txt'
            elif source == 'torchvision':
                from detector import load_cached_detections
                detections = load_cached_detections(sequence, config, cache_dir, max_frames)
                input_path = Path(cache_dir)/name/'manifest.json'
            elif source == 'external' and detections_dir:
                # CSV interno frame,xywh,score; não export MOT de dez colunas.
                input_path = Path(detections_dir)/f'{name}.csv'
                detections = np.loadtxt(input_path, delimiter=',', ndmin=2) if input_path.stat().st_size else np.empty((0, 6))
            else:
                raise ValueError('Fonte inválida ou --detections-dir ausente')
            result, pred = evaluate_sequence(sequence, detections, config, source, max_frames)
            results.append(result)
            save_predictions(output/'predictions'/source/f'{name}.txt', pred)
            with input_path.open('rb') as stream:
                det_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            with (sequence.directory/'gt/gt.txt').open('rb') as stream:
                gt_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            inputs[f'{name}/{source}'] = {'detections_sha256': det_hash, 'gt_sha256': gt_hash}
            ap = 'N/A' if result['ap50'] is None else f"{result['ap50']:.4f}"
            map_value = 'N/A' if result['map_50_95'] is None else f"{result['map_50_95']:.4f}"
            print(f"{name} [{source}] IDF1={result['idf1']:.4f} AP50={ap} "
                  f"mAP={map_value} IDs={result['predicted_identities']}/{result['gt_identities']}", flush=True)
    save_results(results, config, output, inputs)
    from visualization import plot_descolamento
    plot_descolamento(results, output/'descolamento.png', validation_only=True)
    plot_descolamento(results, output/'descolamento_todas.png', validation_only=False)
    return results


def save_results(results, config, output, inputs):
    import scipy
    payload = {'config': config, 'results': results, 'inputs': inputs,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
               'complete_part1': all(any(r['sequence'] == n and r['source'] == s and r['full_sequence']
                                        for r in results)
                                     for n in config['split']['validation'] for s in ['public', 'torchvision'])}
    code_hashes = {}
    for name in ['parte1.py', 'metrics.py', 'geometry.py', 'baseline.py', 'mot_data.py', 'detection_metrics.py']:
        with Path(__file__).with_name(name).open('rb') as stream:
            code_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    payload['code_sha256'] = code_hashes
    (output/'results.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    flat = [{k: v for k, v in r.items() if k != 'ap_by_iou'} for r in results]
    if flat:
        with (output/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate', 'detect', 'benchmark', 'visualize', 'run'])
    parser.add_argument('--config', default='configs/parte1.json')
    parser.add_argument('--root', default='data/MOT17')
    parser.add_argument('--output', default='outputs/parte1')
    parser.add_argument('--cache-dir', default='data/detections/torchvision')
    parser.add_argument('--sources', nargs='+', choices=['public', 'torchvision', 'external'], default=['public'])
    parser.add_argument('--split', choices=['all', 'train', 'validation'], default='all')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--max-frames', type=int, help='Somente depuração; marca avaliação como parcial')
    parser.add_argument('--detections-dir', help='CSV internos para teste de estresse')
    parser.add_argument('--sequence', default='MOT17-09')
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames deve ser positivo')
    config = load_config(args.config)
    status_path = Path(args.output)/'run-status.json'
    def status(value, **extra):
        if args.command != 'run':
            return
        status_path.parent.mkdir(parents=True, exist_ok=True)
        record = {'status': value, 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(status_path)
    try:
        if args.command == 'evaluate':
            evaluate_all(args.root, config, args.output, tuple(args.sources), args.split,
                         args.cache_dir, args.max_frames, args.detections_dir)
        elif args.command == 'visualize':
            from visualization import example_strips
            sequence = Sequence.load(args.root, args.sequence, config['public_detector'])
            example_strips(sequence, Path(args.output)/'predictions/public'/f'{args.sequence}.txt',
                           Path(args.output)/'examples', config)
        else:
            from detector import build_detector, benchmark, extract_detections
            names = config['split']['validation'] if args.command in ('detect', 'run') else [args.sequence]
            sequences = [Sequence.load(args.root, name, config['public_detector']) for name in names]
            # Falhar antes de carregar pesos quando as imagens estão ausentes.
            for sequence in sequences:
                limit = min(sequence.length, args.max_frames or sequence.length) if args.command in ('detect', 'run') else min(35, sequence.length)
                missing = [f for f in range(1, limit+1) if not sequence.image_path(f).exists()]
                if missing:
                    raise FileNotFoundError(f'{sequence.name}: {len(missing)} imagens ausentes. Execute download_mot17.py --package frames --sequences {sequence.name}')
            model, transform, device, versions = build_detector(config, args.device)
            status('detecting', device=device)
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            if args.command == 'benchmark':
                print(json.dumps(benchmark(sequences[0], model, transform, device, output/'benchmark.json'), indent=2))
            else:
                for sequence in sequences:
                    extract_detections(sequence, config, args.cache_dir, model, transform, device, versions, args.max_frames)
                if args.command == 'run':
                    # Comando único de execução e comparação; não treina detector.
                    del model
                    status('evaluating', device=device)
                    evaluate_all(args.root, config, args.output, ('public', 'torchvision'),
                                 'all', args.cache_dir, args.max_frames)
                    payload = json.loads((output/'results.json').read_text(encoding='utf-8'))
                    status('complete' if payload['complete_part1'] else 'partial', device=device)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        status('failed', error=str(error))
        print(f'Erro: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
