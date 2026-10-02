"""GRU causal de movimento e associação própria para caixas MOT17."""
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from geometry import match_iou
from mot_data import Sequence, valid_gt, validate_detections


class MotionGRU(nn.Module):
    """Trilha A com correção residual da Parte 4: caixa + deslocamento."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(4, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, boxes, hidden=None):
        states, hidden = self.gru(boxes, hidden)
        return boxes + self.fc(states), hidden


class MotionRNN(nn.Module):
    """RNN simples com orçamento de parâmetros próximo ao da GRU."""

    def __init__(self, hidden_dim=112):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(4, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, boxes, hidden=None):
        states, hidden = self.rnn(boxes, hidden)
        return boxes + self.fc(states), hidden


class TrajectoryWindows(Dataset):
    """Janelas apenas de tracks GT contíguas; split por vídeo físico."""

    def __init__(self, root, names, detector='FRCNN', steps=16):
        if steps < 2:
            raise ValueError('steps deve ser >= 2')
        windows = []
        for name in names:
            seq = Sequence.load(root, name, detector)
            raw = valid_gt(seq.ground_truth())
            scale = np.array([seq.width, seq.height, seq.width, seq.height], float)
            for gid in np.unique(raw[:, 1]):
                track = raw[raw[:, 1] == gid]
                track = track[np.argsort(track[:, 0])]
                for start in range(len(track) - steps):
                    segment = track[start:start + steps + 1]
                    if np.all(np.diff(segment[:, 0]) == 1):
                        windows.append((segment[:, 2:6] / scale).astype(np.float32))
        if not windows:
            raise ValueError('Nenhuma trajetória GT contígua disponível')
        self.data = torch.from_numpy(np.stack(windows))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        window = self.data[index]
        return window[:-1], window[1:]


def choose_device(requested='auto'):
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device deve ser auto, cpu ou cuda')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA indisponível neste kernel')
    return torch.device('cuda' if requested == 'cuda' or requested == 'auto' and torch.cuda.is_available() else 'cpu')


def train_motion(root='data/MOT17', config=None, checkpoint='checkpoints/temporal_mot17.pt',
                 epochs=8, steps=16, batch_size=256, seed=42, device='auto',
                 max_windows=None, cell='GRU'):
    """Treina só em GT das sequências de desenvolvimento; validação não entra."""
    if config is None:
        from mot_data import load_config
        config = load_config()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    target = choose_device(device)
    dataset = TrajectoryWindows(root, config['split']['train'], config['public_detector'], steps)
    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError('max_windows deve ser positivo')
        indices = np.random.default_rng(seed).choice(len(dataset), size=min(max_windows, len(dataset)), replace=False)
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if cell not in ('GRU', 'RNN'):
        raise ValueError('cell deve ser GRU ou RNN')
    model = (MotionGRU() if cell == 'GRU' else MotionRNN()).to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(target), labels.to(target)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(inputs)
            loss = nn.functional.smooth_l1_loss(predictions, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(inputs)
        mean = total / len(dataset)
        history.append({'epoch': epoch, 'smooth_l1': mean})
        print(f'epoch {epoch}/{epochs}: loss={mean:.6f}', flush=True)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.cpu().state_dict(), 'hidden_dim': model.hidden_dim,
                'prediction': 'residual', 'cell': cell,
                'training': {'sequences': config['split']['train'], 'epochs': epochs,
                             'steps': steps, 'batch_size': batch_size, 'seed': seed,
                             'windows': len(dataset), 'history': history,
                             'detector': config['public_detector']}}, checkpoint)
    return checkpoint, history


def load_motion(checkpoint='checkpoints/temporal_mot17.pt', device='auto'):
    target = choose_device(device)
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if data.get('prediction') != 'residual':
        raise ValueError('Checkpoint incompatível com a GRU residual')
    model = (MotionGRU(data['hidden_dim']) if data.get('cell', 'GRU') == 'GRU'
             else MotionRNN(data['hidden_dim'])).to(target)
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, data['training']


def clip_boxes(boxes, width, height):
    result = np.asarray(boxes, float).reshape(-1, 4).copy()
    result[:, 2:] = np.clip(result[:, 2:], 1, [width, height])
    result[:, :2] = np.clip(result[:, :2], 0, [width, height] - result[:, 2:])
    return result


class TemporalTracker:
    """Uma memória GRU por ID; previsão durante lacunas sem emitir caixas."""

    def __init__(self, model, width, height, threshold=0.3, max_age=8):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.scale = np.array([width, height, width, height], np.float32)
        self.width, self.height = width, height
        self.threshold, self.max_age = threshold, max_age
        self.tracks = {}
        self.next_id = 1
        self.last_frame = 0

    def update(self, frame, boxes):
        if frame <= self.last_frame:
            raise ValueError('Quadros devem ser estritamente crescentes')
        self.last_frame = frame
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if len(boxes):
            boxes = clip_boxes(boxes, self.width, self.height)
        ids = list(self.tracks)
        forecast = [self.tracks[i]['forecast'] for i in ids]
        self.last_forecasts = {tid: box.copy() for tid, box in zip(ids, forecast)}
        pairs = match_iou(forecast, boxes, self.threshold) if ids and len(boxes) else []
        assignment = {j: ids[i] for i, j in pairs}
        updated = {}
        output = []
        for j, box in enumerate(boxes):
            tid = assignment.get(j)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                recurrent = self.model.gru if isinstance(self.model, MotionGRU) else self.model.rnn
                hidden = torch.zeros(recurrent.num_layers, recurrent.hidden_size,
                                     device=self.device)
            else:
                hidden = self.tracks[tid]['hidden']
            updated[tid] = {'input_box': box, 'hidden': hidden, 'missed': 0}
            output.append([frame, tid, *box])
        for tid in ids:
            if tid in updated:
                continue
            track = self.tracks[tid]
            missed = track['missed'] + 1
            if missed <= self.max_age:
                updated[tid] = {'input_box': track['forecast'], 'hidden': track['hidden'],
                                'missed': missed}
        if updated:
            keys = list(updated)
            values = np.stack([updated[i]['input_box'] for i in keys]).astype(np.float32) / self.scale
            inputs = torch.from_numpy(values).to(self.device).unsqueeze(1)
            hidden = torch.stack([updated[i]['hidden'] for i in keys], dim=1)
            with torch.inference_mode():
                next_box, next_hidden = self.model(inputs, hidden)
            forecast = clip_boxes(next_box[:, 0].cpu().numpy() * self.scale,
                                  self.width, self.height)
            self.tracks = {tid: {'forecast': forecast[k], 'hidden': next_hidden[:, k, :].detach(),
                                 'missed': updated[tid]['missed']}
                           for k, tid in enumerate(keys)}
        else:
            self.tracks = {}
        return output


def track_sequence(sequence, detections, model, score_threshold=0.5,
                   association_iou=0.3, max_age=8):
    detections = validate_detections(detections, sequence.length)
    detections = detections[detections[:, 5] >= score_threshold]
    tracker = TemporalTracker(model, sequence.width, sequence.height, association_iou, max_age)
    rows = []
    for frame in range(1, sequence.length + 1):
        rows.extend(tracker.update(frame, detections[detections[:, 0] == frame, 1:5]))
    return np.asarray(rows, float).reshape(-1, 6)
