"""Execute: py -3.12 run_synthetic.py"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from synthetic import generate, corrupt_detections
from baseline import IoUTracker
from metrics import evaluate


def run(scene, seed):
    det = corrupt_detections(scene.gt, seed=seed)
    tracker = IoUTracker()
    predictions = []
    for f in range(1, len(scene.frames)+1):
        predictions.extend(tracker.update(f, det[det[:, 0] == f, 1:5]))
    # Avaliação sintética apenas nas observações com algum pixel visível.
    return evaluate(scene.gt[scene.gt[:, 6] > 0, :6], predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/parte0')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    easy = generate()
    easy_metrics = run(easy, 0)
    if easy_metrics['idf1'] < 0.98:
        raise AssertionError(f'Baseline falhou no piso fácil: {easy_metrics}')
    scene = generate(occlusion=10)
    hidden = scene.gt[(scene.gt[:, 1] == 1) & (scene.gt[:, 6] == 0), 0]
    assert len(hidden) == 10, hidden
    frames = [Image.fromarray(f).resize((384, 384)) for f in scene.frames]
    frames[0].save(out/'oclusao.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
    samples = [int(hidden[0])-2, int(hidden[0])-1, int(hidden[-1])-1, int(hidden[-1])]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, t in zip(axes, samples):
        ax.imshow(scene.frames[t])
        vis = scene.gt[(scene.gt[:, 0] == t+1) & (scene.gt[:, 1] == 1), 6][0]
        ax.set_title(f'Quadro {t+1}: vis. {vis:.0%}')
        ax.axis('off')
    fig.suptitle('Objeto 1 em (64, 96): desaparece por 10 quadros e retorna')
    fig.tight_layout()
    fig.savefig(out/'oclusao.png', dpi=150)
    plt.close(fig)
    experiments = {}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (parameter, values, label) in zip(axes, [
        ('n_objects', [5, 10, 15], 'Número de objetos'),
        ('speed', [0.3, 2, 5, 9], 'Velocidade (pixels/quadro)'),
        ('occlusion', [0, 4, 10, 20], 'Oclusão (quadros)')]):
        groups = [[run(generate(**{parameter: value}, seed=s), s) for s in range(3)] for value in values]
        experiments[parameter] = dict(values=values, metrics=groups)
        scores = np.array([[m['idf1'] for m in group] for group in groups])
        ax.errorbar(values, scores.mean(1), yerr=scores.std(1, ddof=1), marker='o', capsize=3)
        ax.set(xlabel=label, ylabel='IDF1', ylim=(0, 1.05))
        ax.grid(alpha=0.25)
    fig.suptitle('Baseline IoU: média e desvio amostral de 3 seeds')
    fig.tight_layout()
    fig.savefig(out/'baseline_dificuldade.png', dpi=150)
    plt.close(fig)
    result = dict(easy=easy_metrics, fully_hidden_frames=hidden.astype(int).tolist(), experiments=experiments)
    (out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(easy_metrics, indent=2))
    print(f'Artefatos: {out.resolve()}')


if __name__ == '__main__':
    main()
