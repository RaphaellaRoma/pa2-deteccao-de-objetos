"""Figuras de detecção/rastreamento e exemplos com cores determinísticas por ID."""
from pathlib import Path
import colorsys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from geometry import match_iou
from mot_data import load_predictions, valid_gt


def plot_descolamento(results, output, validation_only=True):
    rows = [r for r in results if not validation_only or r['split'] == 'validation']
    if not rows:
        return None
    density = {r['sequence']: r['density'] for r in rows}
    names = sorted(density, key=lambda n: (density[n], n))
    sources = sorted({r['source'] for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(max(9, len(names)*1.4), 7), sharex=True)
    switch_axis = axes[1].twinx()
    colors = {'public': '#1764ab', 'torchvision': '#df7517', 'external': '#32934a'}
    for source in sources:
        lookup = {r['sequence']: r for r in rows if r['source'] == source}
        def values(key):
            return [lookup[n][key] if n in lookup and lookup[n][key] is not None else np.nan for n in names]
        color = colors.get(source)
        axes[0].plot(values('map_50_95'), 'o--', color=color, label=f'{source}: mAP@0,50:0,95')
        axes[0].plot(values('idf1'), 's-', color=color, label=f'{source}: IDF1')
        axes[1].plot(values('identity_ratio'), 'o-', color=color, label=f'{source}: IDs previstos/GT')
        switch_axis.plot(values('switches_per_gt_identity'), 'x:', color=color, label=f'{source}: switches/ID GT')
    axes[0].set(ylabel='mAP / IDF1', ylim=(0, 1.05))
    axes[1].set(ylabel='IDs previstos / IDs GT')
    switch_axis.set_ylabel('ID switches / ID GT')
    axes[1].axhline(1, color='gray', linewidth=1, linestyle='--')
    axes[1].set_xticks(range(len(names)), [f'{n}\ndensidade={density[n]:.1f}' for n in names])
    axes[1].set_xlabel('Sequências ordenadas pela densidade média de pedestres válidos')
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=9)
    h, l = axes[1].get_legend_handles_labels()
    h2, l2 = switch_axis.get_legend_handles_labels()
    axes[1].legend(h+h2, l+l2, fontsize=8)
    partial = any(not r['full_sequence'] for r in rows)
    subtitle = 'DEPURAÇÃO: sequências parciais' if partial else 'Sequências completas'
    if len(sources) == 1:
        subtitle += ' | comparação entre detectores pendente'
    fig.suptitle(f'Parte 1 - descolamento entre detecção e identidade\n{subtitle}')
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def identity_color(identity):
    return colorsys.hsv_to_rgb((int(identity)*0.61803398875) % 1, 0.85, 1.0)


def draw_boxes(ax, rows):
    for _, identity, x, y, w, h in rows:
        color = identity_color(identity)
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.4))
        ax.text(x, y, str(int(identity)), color='black', fontsize=7, clip_on=True,
                bbox={'facecolor': color, 'alpha': 0.8, 'pad': 1})


def example_strips(sequence, predictions_path, output, config):
    """Encontra um ID mantido, uma troca e um retorno com ID novo, se existirem."""
    gt = valid_gt(sequence.ground_truth())
    pred = load_predictions(predictions_path)
    previous = {}
    examples = {}
    for frame in range(1, sequence.length+1):
        g, p = gt[gt[:, 0] == frame], pred[pred[:, 0] == frame]
        for i, j in match_iou(g[:, 2:], p[:, 2:], config['evaluation_iou']):
            identity, predicted = int(g[i, 1]), int(p[j, 1])
            if identity in previous:
                old_frame, old_id = previous[identity]
                kind = 'mantido' if old_id == predicted and frame == old_frame+1 else 'troca' if old_id != predicted and frame == old_frame+1 else 'retorno_id_novo' if old_id != predicted else None
                if kind and kind not in examples:
                    examples[kind] = {'gt_id': identity, 'before': old_frame, 'after': frame,
                                      'prediction_before': old_id, 'prediction_after': predicted}
            previous[identity] = frame, predicted
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for kind, event in examples.items():
        frames = sorted(set([max(1, event['before']-1), event['before'], event['after'], min(sequence.length, event['after']+1)]))
        fig, axes = plt.subplots(2, len(frames), figsize=(4*len(frames), 6.5), squeeze=False)
        for column, frame in enumerate(frames):
            with Image.open(sequence.image_path(frame)) as image:
                pixels = np.asarray(image.convert('RGB'))
            for row, (label, values) in enumerate([('GT', gt), ('Baseline', pred)]):
                ax = axes[row, column]
                ax.imshow(pixels)
                draw_boxes(ax, values[values[:, 0] == frame])
                ax.set_title(f'{label} - quadro {frame}')
                ax.axis('off')
        fig.suptitle(f"{sequence.name}: {kind} | GT {event['gt_id']} | "
                     f"ID previsto {event['prediction_before']} -> {event['prediction_after']}")
        fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0)
        fig.savefig(output/f'{sequence.name}-{kind}.png', dpi=130)
        plt.close(fig)
    import json
    (output/f'{sequence.name}-events.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
    return examples
