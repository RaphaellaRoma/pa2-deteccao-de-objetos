"""AP próprio de uma classe, com matching espacial independente em cada quadro."""
import numpy as np
from geometry import iou_matrix


def average_precision(gt, detections, threshold=0.5):
    """GT: frame,id,xywh; det: frame,xywh,score. AP interpolado em 101 recalls."""
    gt = np.asarray(gt, float).reshape(-1, 6)
    det = np.asarray(detections, float).reshape(-1, 6)
    if not len(gt):
        return None
    # Ordenação global por confiança; matching só é permitido no mesmo quadro.
    order = np.argsort(-det[:, 5], kind='stable')
    groups = {f: gt[gt[:, 0] == f, 2:6] for f in np.unique(gt[:, 0])}
    used = {f: np.zeros(len(boxes), dtype=bool) for f, boxes in groups.items()}
    tp = np.zeros(len(det))
    for rank, index in enumerate(order):
        frame = det[index, 0]
        if frame not in groups:
            continue
        overlap = iou_matrix(det[index:index+1, 1:5], groups[frame])[0]
        candidates = np.where(~used[frame], overlap, -1)
        best = int(np.argmax(candidates))
        if candidates[best] >= threshold:
            used[frame][best] = True
            tp[rank] = 1
    recall = np.cumsum(tp)/len(gt)
    precision = np.cumsum(tp)/np.arange(1, len(det)+1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    samples = []
    for r in np.linspace(0, 1, 101):
        values = envelope[recall >= r]
        samples.append(float(values[0]) if len(values) else 0.0)
    return float(np.mean(samples))


def evaluate_detection(gt, detections):
    thresholds = np.linspace(0.5, 0.95, 10)
    values = [average_precision(gt, detections, float(t)) for t in thresholds]
    return {'ap50': values[0], 'map_50_95': float(np.mean(values)) if values[0] is not None else None,
            'ap_by_iou': {f'{t:.2f}': v for t, v in zip(thresholds, values)}}
