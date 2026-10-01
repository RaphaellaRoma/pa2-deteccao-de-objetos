"""Métricas próprias. Entrada: linhas [frame, id, x, y, w, h].

IDF1: atribuição global de identidades com contagem de quadros espacialmente
compatíveis (não depende do matching local). IDSW: troca da última identidade
associada, inclusive depois de lacunas. Fragmentação: matched -> unmatched ->
matched dentro das observações GT. Ausência de uma linha GT não conta como FN.
Não implementa ainda o protocolo de regiões ignoradas do MOTChallenge.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from geometry import iou_matrix, match_iou


def _rows(value):
    a = np.asarray(value, dtype=float).reshape(-1, 6)
    if not np.isfinite(a).all() or np.any(a[:, 4:] <= 0):
        raise ValueError('Caixas devem ser finitas e ter área positiva')
    if np.any(a[:, :2] != np.floor(a[:, :2])):
        raise ValueError('Frames e IDs devem ser inteiros')
    if len(np.unique(a[:, :2], axis=0)) != len(a):
        raise ValueError('Uma identidade só pode aparecer uma vez por quadro')
    return a


def evaluate(gt, pred, threshold=0.5):
    gt, pred = _rows(gt), _rows(pred)
    gids, pids = np.unique(gt[:, 1]), np.unique(pred[:, 1])
    gi = {x: i for i, x in enumerate(gids)}
    pi = {x: i for i, x in enumerate(pids)}
    compatibility = np.zeros((len(gids), len(pids)), dtype=int)
    previous, seen, gap = {}, set(), set()
    switches = fragments = 0
    for frame in np.union1d(gt[:, 0], pred[:, 0]):
        g, p = gt[gt[:, 0] == frame], pred[pred[:, 0] == frame]
        for i, j in zip(*np.where(iou_matrix(g[:, 2:], p[:, 2:]) >= threshold)):
            compatibility[gi[g[i, 1]], pi[p[j, 1]]] += 1
        matches = dict(match_iou(g[:, 2:], p[:, 2:], threshold))
        for i, row in enumerate(g):
            identity = row[1]
            if i not in matches:
                if identity in seen:
                    gap.add(identity)
                continue
            predicted = p[matches[i], 1]
            switches += int(identity in previous and previous[identity] != predicted)
            fragments += int(identity in gap)
            gap.discard(identity)
            seen.add(identity)
            previous[identity] = predicted
    r, c = linear_sum_assignment(-compatibility)
    idtp = int(compatibility[r, c].sum())
    return dict(idf1=2 * idtp / (len(gt) + len(pred)) if len(gt) + len(pred) else 1.0,
                idtp=idtp, idfp=len(pred)-idtp, idfn=len(gt)-idtp,
                id_switches=switches, fragmentations=fragments,
                gt_identities=len(gids), predicted_identities=len(pids),
                unique_count_error=abs(len(pids)-len(gids)))
