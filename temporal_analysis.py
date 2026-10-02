"""Horizonte, galeria de falhas e correção observada no MOT17."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image

from geometry import iou_matrix, match_iou
from metrics import evaluate
from mot_data import Sequence, evaluation_mask, load_config, load_predictions, valid_gt
from temporal import TemporalTracker, load_motion, track_sequence
from visualization import identity_color


def gradient_horizon(model, sequence, steps=16, sample_count=12):
    """dL_t/dh_(t-k), k=1..T-1, em janelas GT reais contíguas."""
    raw = valid_gt(sequence.ground_truth())
    scale = torch.tensor([sequence.width, sequence.height, sequence.width, sequence.height],
                         dtype=torch.float32, device=next(model.parameters()).device)
    windows = []
    for gid in np.unique(raw[:, 1]):
        track = raw[raw[:, 1] == gid]
        track = track[np.argsort(track[:, 0])]
        for start in range(0, len(track) - steps, max(1, steps // 2)):
            segment = track[start:start + steps + 1]
            if np.all(np.diff(segment[:, 0]) == 1):
                windows.append(segment[:, 2:6].astype(np.float32))
                if len(windows) == sample_count:
                    break
        if len(windows) == sample_count:
            break
    if not windows:
        raise ValueError('Sem janelas GT contíguas para medir gradientes')
    curves = []
    model.eval()
    for window in windows:
        model.zero_grad(set_to_none=True)
        data = torch.from_numpy(window).to(scale.device) / scale
        state = None
        states = []
        for t in range(steps):
            pred, state = model(data[t:t + 1].reshape(1, 1, 4), state)
            state.retain_grad()
            states.append(state)
        loss = torch.nn.functional.smooth_l1_loss(pred[0, 0], data[steps])
        loss.backward()
        curves.append([float(states[-1 - lag].grad.norm().item()) for lag in range(1, steps)])
    return np.mean(curves, axis=0)


def matched_ids(gt, predictions, threshold=0.5):
    """Mapa (frame, GT-ID)->pred-ID, apenas para diagnóstico retrospectivo."""
    mapping = {}
    for frame in np.unique(gt[:, 0]):
        g = gt[gt[:, 0] == frame]
        p = predictions[predictions[:, 0] == frame]
        for i, j in match_iou(g[:, 2:6], p[:, 2:6], threshold):
            mapping[(int(frame), int(g[i, 1]))] = int(p[j, 1])
    return mapping


def occlusion_episodes(sequence):
    """Runs contíguos de visibility=0, cercados por observações válidas."""
    raw = sequence.ground_truth()
    raw = raw[(raw[:, 6] > 0) & (raw[:, 7] == 1)]
    episodes = []
    for gid in np.unique(raw[:, 1]):
        rows = raw[raw[:, 1] == gid]
        rows = rows[np.argsort(rows[:, 0])]
        hidden = rows[:, 8] <= 0
        starts = np.flatnonzero(hidden & ~np.r_[False, hidden[:-1]])
        for start in starts:
            end = start
            while end + 1 < len(rows) and hidden[end + 1] and rows[end + 1, 0] == rows[end, 0] + 1:
                end += 1
            if start and end + 1 < len(rows) and rows[start, 0] == rows[start - 1, 0] + 1 \
                    and rows[end + 1, 0] == rows[end, 0] + 1 and rows[end + 1, 8] > 0:
                episodes.append({'sequence': sequence.name, 'gt_id': int(gid),
                                 'before': int(rows[start - 1, 0]),
                                 'after': int(rows[end + 1, 0]), 'duration': int(end - start + 1)})
    return episodes


def survival(episodes, matches):
    measured = []
    for episode in episodes:
        key = episode['gt_id']
        before = matches.get((episode['before'], key))
        after = matches.get((episode['after'], key))
        if before is not None:
            measured.append({**episode, 'old_id': before, 'new_id': after,
                             'survived': before == after})
    return measured


def switch_events(gt, pred):
    mapped = matched_ids(gt, pred)
    observations = defaultdict(list)
    for (frame, gid), pid in mapped.items():
        observations[gid].append((frame, pid))
    events = []
    for gid, rows in observations.items():
        rows.sort()
        for (before, old), (after, new) in zip(rows, rows[1:]):
            if old != new:
                events.append({'gt_id': gid, 'before': before, 'after': after,
                               'old_id': old, 'new_id': new, 'gap': after - before})
    return events


def failure_strip(sequence, detections, model, pred, event, output, max_age):
    frames = sorted({event['before'], min(event['before'] + 1, event['after']),
                     max(event['before'], event['after'] - 1), event['after']})
    selected = set(frames)
    tracker = TemporalTracker(model, sequence.width, sequence.height, max_age=max_age)
    forecasts = {}
    for frame in range(1, max(frames) + 1):
        boxes = detections[detections[:, 0] == frame, 1:5]
        tracker.update(frame, boxes)
        if frame in selected:
            forecasts[frame] = tracker.last_forecasts
    gt = valid_gt(sequence.ground_truth())
    fig, axes = plt.subplots(2, len(frames), figsize=(4 * len(frames), 6), squeeze=False)
    focus = event['gt_id']
    for col, frame in enumerate(frames):
        with Image.open(sequence.image_path(frame)) as image:
            pixels = np.asarray(image.convert('RGB'))
        for row, (label, values, wanted) in enumerate([
                ('GT', gt, focus), ('GRU', pred, None)]):
            ax = axes[row, col]
            ax.imshow(pixels)
            for f, ident, x, y, w, h in values[values[:, 0] == frame]:
                if wanted is not None and int(ident) != wanted:
                    continue
                if wanted is None and int(ident) not in (event['old_id'], event['new_id']):
                    continue
                color = identity_color(ident)
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2))
                ax.text(x, y, str(int(ident)), fontsize=8, color='black',
                        bbox={'facecolor': color, 'alpha': .8, 'pad': 1})
            if row == 1 and event['old_id'] in forecasts[frame]:
                x, y, w, h = forecasts[frame][event['old_id']]
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='deepskyblue',
                                       linewidth=2, linestyle='--'))
            ax.set_title(f'{label} · quadro {frame}')
            ax.axis('off')
    fig.suptitle(f"{sequence.name}: GT {focus}, ID {event['old_id']} → {event['new_id']}; "
                 f"intervalo {event['gap']} quadros\n"
                 'Azul tracejado: caixa prevista pela GRU antes da associação')
    fig.tight_layout(rect=(0, 0, 1, .9))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def analyze(root='data/MOT17', config_path='configs/parte1.json',
            checkpoint='checkpoints/temporal_mot17.pt',
            rnn_checkpoint='checkpoints/temporal_rnn_mot17.pt',
            output='outputs/temporal_analysis', device='auto'):
    config = load_config(config_path)
    model, training = load_motion(checkpoint, device)
    rnn, rnn_training = load_motion(rnn_checkpoint, device)
    if training['steps'] != rnn_training['steps'] or training['sequences'] != rnn_training['sequences']:
        raise ValueError('RNN e GRU precisam usar mesmo split e janela')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    first = Sequence.load(root, config['split']['validation'][0], config['public_detector'])
    steps = training['steps']
    curves = {'RNN': gradient_horizon(rnn, first, steps),
              'GRU': gradient_horizon(model, first, steps)}
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in curves.items():
        ax.plot(range(1, steps), values, marker='o', label=name)
    ax.set(xlabel='Atraso k (quadros)', ylabel='Norma média de ∂L_t/∂h_(t-k)',
           title='Horizonte analítico — MOT17-02, modelos treinados')
    ax.set_yscale('log')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'gradientes.png', dpi=160)
    plt.close(fig)
    episodes_all, survived = [], {'age2': [], 'age8': []}
    comparisons = []
    gallery = []
    for name in config['split']['validation']:
        sequence = Sequence.load(root, name, config['public_detector'])
        detections = sequence.public_detections()
        detections = detections[detections[:, 5] >= config['score_threshold']]
        gt = valid_gt(sequence.ground_truth())
        episodes = occlusion_episodes(sequence)
        episodes_all.extend(episodes)
        predictions = {}
        for age in (2, 8):
            pred = track_sequence(sequence, detections, model,
                                  config['score_threshold'], config['association_iou'], age)
            predictions[age] = pred
            mask = evaluation_mask(sequence.ground_truth(), pred[:, 0], pred[:, 2:6], config['ignore_iou'])
            metrics = evaluate(gt, pred[mask], config['evaluation_iou'])
            comparisons.append({'sequence': name, 'max_age': age, **metrics})
            survived[f'age{age}'].extend(survival(episodes, matched_ids(gt, pred)))
        events = switch_events(gt, predictions[8])
        if events:
            # Prefer an actual ID change near an occlusion, with a readable strip.
            best = sorted(events, key=lambda e: (not 2 <= e['gap'] <= 12,
                        -min(e['gap'], 12), e['before']))[0]
            gallery.append({'sequence': name, **best})
            failure_strip(sequence, detections, model, predictions[8], best,
                          out/f'falha-{name}.png', 8)
        print(f'{name}: episódios de visibility=0: {len(episodes)}, '
              f'trocas: {len(events)}', flush=True)
    labels = ['1–2', '3–5', '6–10', '11–20', '>20']
    def bucket(duration):
        return 0 if duration <= 2 else 1 if duration <= 5 else 2 if duration <= 10 else 3 if duration <= 20 else 4
    counts = [sum(bucket(e['duration']) == i for e in episodes_all) for i in range(5)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(5)
    ax.bar(xs, counts, width=.55, alpha=.45, label='Oclusões GT (todos os episódios)')
    axis = ax.twinx()
    for age, color in [('age2', '#bb642f'), ('age8', '#1764ab')]:
        group = survived[age]
        rates = [np.mean([e['survived'] for e in group if bucket(e['duration']) == i])
                 if any(bucket(e['duration']) == i for e in group) else np.nan for i in range(5)]
        axis.scatter(xs + (-.08 if age == 'age2' else .08), rates, s=65,
                     color=color, label=f'Sobrevivência max_age={age[-1]}')
    eligible = [sum(bucket(e['duration']) == i for e in survived['age8']) for i in range(5)]
    for x, n in zip(xs, eligible):
        axis.text(x, .04, f'n={n}', ha='center', fontsize=8)
    axis.set(ylim=(0, 1.05), ylabel='Fração que manteve ID')
    axis.legend(loc='upper right', fontsize=8)
    ax.set(xticks=xs, xticklabels=labels, xlabel='Duração da oclusão (quadros)',
           ylabel='Quantidade', title='Oclusões GT e sobrevivência empírica — validação MOT17')
    ax.grid(alpha=.25, axis='y')
    fig.tight_layout()
    fig.savefig(out/'sobrevivencia.png', dpi=160)
    plt.close(fig)
    summary = {'training': training, 'rnn_training': rnn_training,
               'gradient_norms': {k: v.tolist() for k, v in curves.items()},
               'occlusion_episodes': episodes_all, 'survival': survived,
               'correction_max_age_2_to_8': comparisons, 'failure_gallery': gallery}
    (out/'results.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/MOT17')
    parser.add_argument('--config', default='configs/parte1.json')
    parser.add_argument('--checkpoint', default='checkpoints/temporal_mot17.pt')
    parser.add_argument('--rnn-checkpoint', default='checkpoints/temporal_rnn_mot17.pt')
    parser.add_argument('--output', default='outputs/temporal_analysis')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = parser.parse_args()
    analyze(a.root, a.config, a.checkpoint, a.rnn_checkpoint, a.output, a.device)
