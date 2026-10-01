"""Geometria própria: caixas no formato [left, top, width, height]."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    inter = np.maximum(hi - lo, 0).prod(axis=2)
    union = a[:, 2:].prod(axis=1)[:, None] + b[:, 2:].prod(axis=1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_iou(a, b, threshold=0.5):
    """Maximiza primeiro quantidade de pares válidos, depois soma dos IoUs."""
    if not 0 < threshold <= 1:
        raise ValueError('threshold deve estar em (0, 1]')
    overlap = iou_matrix(a, b)
    if not overlap.size:
        return []
    valid = overlap >= threshold
    bonus = min(overlap.shape) + 1
    rows, cols = linear_sum_assignment(-(valid * (bonus + overlap)))
    return [(int(i), int(j)) for i, j in zip(rows, cols) if valid[i, j]]


def nms(boxes, scores, threshold=0.5):
    """NMS guloso próprio, com desempate estável."""
    boxes = np.asarray(boxes).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores), kind='stable')
    keep = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        order = rest[iou_matrix(boxes[i:i+1], boxes[rest])[0] <= threshold]
    return keep
