"""Vídeos com elipses rasterizadas em ordem de profundidade e GT de visibilidade."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Scene:
    frames: np.ndarray
    gt: np.ndarray  # frame, id, x, y, w, h, visibility (inclui objetos ocultos)


def generate(n_objects=5, speed=0.3, occlusion=0, n_frames=48,
             noise=2.0, contrast=1.0, seed=0):
    if not 5 <= n_objects <= 15 or not 30 <= n_frames <= 60:
        raise ValueError('Use 5 a 15 objetos e 30 a 60 quadros')
    if speed < 0 or not 0 <= occlusion <= n_frames-4:
        raise ValueError('Velocidade ou duração de oclusão inválida')
    rng = np.random.default_rng(seed)
    size = 128
    yy, xx = np.mgrid[:size, :size]
    centers = np.array([(16+24*(i % 5), 20+32*(i//5)) for i in range(n_objects)], float)
    radii = rng.uniform(4, 7, (n_objects, 2))
    velocity = rng.normal(size=(n_objects, 2))
    velocity *= speed / np.maximum(np.linalg.norm(velocity, axis=1, keepdims=True), 1e-8)
    colors = rng.integers(60, 245, (n_objects, 3))
    start = (n_frames-occlusion)//2
    frames, gt = [], []
    for t in range(n_frames):
        # Reflexão analítica nas bordas, inclusive para velocidades altas.
        span = size-2*radii-2
        phase = np.mod(centers-radii-1+velocity*t, 2*span)
        pos = radii+1+np.minimum(phase, 2*span-phase)
        rad = radii.copy()
        if occlusion:
            # Objeto 2 em primeiro plano cobre de fato todos os pixels do 1.
            # Seu afastamento antes/depois produz exatamente N quadros ocultos.
            pos[0] = [64, 96]
            rad[0] = [4, 4]
            rad[1] = [9, 9]
            if t < start:
                distance = -min(28, 10+4*(start-1-t))
            elif t < start+occlusion:
                distance = 0 if occlusion == 1 else -4+8*(t-start)/(occlusion-1)
            else:
                distance = min(28, 10+4*(t-start-occlusion))
            pos[1] = pos[0] + [distance, 0]
        masks = [((xx-p[0])/r[0])**2 + ((yy-p[1])/r[1])**2 <= 1
                 for p, r in zip(pos, rad)]
        ownership = np.full((size, size), -1)
        canvas = np.full((size, size, 3), 25.0)
        for i, mask in enumerate(masks):
            ownership[mask] = i
            canvas[mask] = 25+contrast*(colors[i]-25)
        for i, (p, r, mask) in enumerate(zip(pos, rad, masks)):
            visible = np.count_nonzero(ownership == i)/np.count_nonzero(mask)
            gt.append([t+1, i+1, *(p-r), *(2*r), visible])
        frames.append(np.clip(canvas+rng.normal(0, noise, canvas.shape), 0, 255).astype(np.uint8))
    return Scene(np.array(frames), np.array(gt))


def corrupt_detections(gt, drop=0.0, jitter=0.0, false_positives=0.0, seed=0):
    """FP ~ Poisson por quadro; jitter em pixels; sem acesso a IDs na saída."""
    if not 0 <= drop <= 1 or jitter < 0 or false_positives < 0:
        raise ValueError('Parâmetros de corrupção inválidos')
    rng = np.random.default_rng(seed)
    output = []
    for frame in np.unique(gt[:, 0]):
        rows = gt[(gt[:, 0] == frame) & (gt[:, 6] > 0)]
        for row in rows:
            if rng.random() < drop:
                continue
            box = row[2:6] + rng.normal(0, jitter, 4)
            box[:2] = np.clip(box[:2], 0, 127)
            box[2:] = np.minimum(np.maximum(box[2:], 1), 128-box[:2])
            output.append([frame, *box, 0.9])
        for _ in range(rng.poisson(false_positives)):
            wh = rng.uniform(5, 15, 2)
            xy = rng.uniform(0, 128-wh)
            output.append([frame, *xy, *wh, 0.5])
    return np.array(output).reshape(-1, 6)
