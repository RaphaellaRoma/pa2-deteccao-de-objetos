"""Contrato MOT: internamente xywh 0-based, quadros e IDs 1-based."""
from dataclasses import dataclass
import configparser
import json
import re
from pathlib import Path
import numpy as np
from geometry import match_iou


def load_config(path='configs/parte1.json'):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    train = config['split']['train']
    val = config['split']['validation']
    if len(set(train+val)) != len(train+val):
        raise ValueError('Split repetido ou vazamento entre treino e validação')
    if not train or not val:
        raise ValueError('Treino e validação devem conter sequências inteiras')
    if any(not re.fullmatch(r'MOT17-\d{2}', name) for name in train+val):
        raise ValueError('Split deve usar nomes físicos MOT17-XX, sem sufixo de detector')
    if not isinstance(config['max_age'], int) or config['max_age'] < 0:
        raise ValueError('max_age deve ser inteiro não negativo')
    for key in ('association_iou', 'evaluation_iou', 'ignore_iou'):
        if not 0 < config[key] <= 1:
            raise ValueError(f'{key} inválido')
    if not 0 <= config['cache_score_threshold'] <= config['score_threshold'] <= 1:
        raise ValueError('Limiar de confiança/cache inválido')
    return config


def read_table(path, min_columns=6):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Arquivo ausente: {path}. Execute download_mot17.py primeiro.')
    if not path.read_text(encoding='utf-8').strip():
        return np.empty((0, min_columns))
    rows = np.loadtxt(path, delimiter=',', ndmin=2)
    if rows.shape[1] < min_columns or not np.isfinite(rows).all():
        raise ValueError(f'Tabela MOT inválida: {path}')
    if np.any(rows[:, 0] < 1) or np.any(rows[:, 0] != np.floor(rows[:, 0])):
        raise ValueError('Quadros MOT devem ser inteiros positivos')
    if np.any(rows[:, 4:6] <= 0):
        raise ValueError('Largura e altura devem ser positivas')
    return rows


@dataclass
class Sequence:
    name: str
    directory: Path
    length: int
    width: int
    height: int
    fps: float
    image_dir: str
    image_ext: str

    @classmethod
    def load(cls, root, name, detector='FRCNN'):
        directory = Path(root) / 'train' / f'{name}-{detector}'
        ini = configparser.ConfigParser()
        if not ini.read(directory/'seqinfo.ini'):
            raise FileNotFoundError(f'seqinfo.ini ausente em {directory}')
        s = ini['Sequence']
        return cls(name, directory, int(s['seqLength']), int(s['imWidth']),
                   int(s['imHeight']), float(s['frameRate']), s['imDir'], s['imExt'])

    def ground_truth(self):
        rows = read_table(self.directory/'gt/gt.txt', 9)[:, :9].copy()
        rows[:, 2:4] -= 1
        if len(rows) and rows[:, 0].max() > self.length:
            raise ValueError('GT fora dos limites da sequência')
        return rows

    def public_detections(self):
        raw = read_table(self.directory/'det/det.txt', 7)
        rows = raw[:, [0, 2, 3, 4, 5, 6]].copy()
        rows[:, 1:3] -= 1
        return validate_detections(rows, self.length)

    def image_path(self, frame):
        if not 1 <= frame <= self.length:
            raise ValueError('Quadro fora da sequência')
        return self.directory / self.image_dir / f'{frame:06d}{self.image_ext}'


def validate_detections(rows, length=None):
    rows = np.asarray(rows, dtype=float).reshape(-1, 6)
    if not np.isfinite(rows).all() or np.any(rows[:, 3:5] <= 0):
        raise ValueError('Detecções não finitas ou caixas sem área')
    if np.any(rows[:, 0] < 1) or np.any(rows[:, 0] != np.floor(rows[:, 0])):
        raise ValueError('Quadros de detecções inválidos')
    if length is not None and np.any(rows[:, 0] > length):
        raise ValueError('Detecções fora dos limites da sequência')
    return rows


def save_predictions(path, rows):
    """Exporta dez colunas MOT, com posições 1-based; entrada interna de seis colunas."""
    rows = np.asarray(rows, dtype=float).reshape(-1, 6).copy()
    rows[:, 2:4] += 1
    extra = np.tile([1, -1, -1, -1], (len(rows), 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([rows, extra]), delimiter=',', fmt='%.8g')


def load_predictions(path):
    rows = read_table(path, 6)[:, :6].copy()
    rows[:, 2:4] -= 1
    return rows


def valid_gt(raw):
    """Não descarta observações de baixa visibilidade, inclusive visibility=0."""
    return raw[(raw[:, 6] > 0) & (raw[:, 7] == 1), :6]


def evaluation_mask(raw_gt, frames, boxes, threshold=0.5):
    """Mantém matches de pedestres; ignora apenas sobras casadas a ignorados.

    Protocolo próprio pa2-pedestrians-v1, não um clone do avaliador MOTChallenge.
    Este filtro nunca é aplicado às entradas do rastreador.
    """
    frames = np.asarray(frames)
    boxes = np.asarray(boxes).reshape(-1, 4)
    keep = np.ones(len(boxes), dtype=bool)
    for frame in np.unique(frames):
        idx = np.flatnonzero(frames == frame)
        gt = raw_gt[raw_gt[:, 0] == frame]
        targets = gt[(gt[:, 6] > 0) & (gt[:, 7] == 1)]
        ignored = gt[(gt[:, 6] == 0) | np.isin(gt[:, 7], [2, 7, 8, 12])]
        protected = {j for _, j in match_iou(targets[:, 2:6], boxes[idx], threshold)}
        remaining = [j for j in range(len(idx)) if j not in protected]
        for _, j in match_iou(ignored[:, 2:6], boxes[idx[remaining]], threshold):
            keep[idx[remaining[j]]] = False
    return keep
