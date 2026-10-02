"""Estresse das deteccoes congeladas, sem GT no gerador e sem retreino."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from metrics import evaluate
from mot_data import (Sequence, load_config, validate_detections, save_predictions,
                      load_predictions, valid_gt, evaluation_mask)
from parte1 import evaluate_sequence


def corrupt_detections(detections, length, width, height, parameters, seed):
    """Ruido relativo a w/h em centro e tamanho; FP uniformes no quadro.

    Recebe somente deteccoes e metadados. Scores reais preservados; FP recebem
    scores amostrados das deteccoes de entrada. Nenhum acesso ao ground truth.
    """
    original = validate_detections(detections, length)
    p = parameters['drop_probability']
    noise = parameters['box_noise_fraction']
    rate = parameters['false_positives_per_frame']
    if not 0 <= p <= 1 or not np.isfinite([noise, rate]).all() or min(noise, rate) < 0:
        raise ValueError('Parametros de degradacao invalidos')
    if min(length, width, height) <= 0:
        raise ValueError('Dimensoes e duracao devem ser positivas')
    if p == noise == rate == 0:
        return original.copy()
    rng = np.random.default_rng(seed)
    kept = original[rng.random(len(original)) >= p].copy()
    if len(kept):
        sizes = kept[:, 3:5].copy()
        centers = kept[:, 1:3] + sizes / 2
        centers += rng.normal(size=sizes.shape) * noise * sizes
        sizes *= np.exp(rng.normal(size=sizes.shape) * noise)
        sizes = np.clip(sizes, 1, [width, height])
        kept[:, 1:3] = np.clip(centers - sizes / 2, 0, [width, height] - sizes)
        kept[:, 3:5] = sizes
    counts = rng.poisson(rate, length)
    frames = np.repeat(np.arange(1, length + 1), counts)
    if len(frames):
        # Empirical box sizes/scores; no GT-dependent placement or labelling.
        if len(original):
            sampled = original[rng.integers(len(original), size=len(frames))]
            sizes = np.clip(sampled[:, 3:5], 1, [width, height])
            scores = sampled[:, 5]
        else:
            sizes = np.tile([max(1, width * .03), max(1, height * .1)], (len(frames), 1))
            scores = np.full(len(frames), .9)
        positions = rng.random(sizes.shape) * ([width, height] - sizes)
        fp = np.column_stack([frames, positions, sizes, scores])
        kept = np.vstack([kept, fp])
    return validate_detections(kept[np.argsort(kept[:, 0], kind='stable')], length)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plot_results(rows, path, levels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model in sorted({r['model'] for r in rows}):
        for ax, metric, label in zip(axes, ['map_50_95', 'idf1'], ['mAP 0,50:0,95', 'IDF1']):
            # Equal-weight sequence mean within each seed, then mean/std over seeds.
            means, deviations = [], []
            for level in levels:
                seed_means = [np.mean([r[metric] for r in rows if r['model'] == model and
                                      r['level'] == level and r['seed'] == seed])
                              for seed in sorted({r['seed'] for r in rows if r['level'] == level})]
                means.append(np.mean(seed_means))
                deviations.append(np.std(seed_means))
            ax.errorbar(range(len(levels)), means, yerr=deviations, marker='o', capsize=4, label=model)
            ax.set(xticks=range(len(levels)), xticklabels=levels, ylabel=label, ylim=(0, 1))
            ax.grid(alpha=.25)
            ax.legend()
    fig.suptitle('Estresse: media por sequencia; barras = desvio entre seeds\n'
                 + ('Comparacao com modelo temporal pendente' if not any(r['model'] == 'temporal' for r in rows)
                    else 'Mesmas deteccoes para baseline e temporal'))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(root, output, config_path, stress_path, temporal_predictions=None):
    config = load_config(config_path)
    stress = json.loads(Path(stress_path).read_text(encoding='utf-8'))
    if not stress['seeds'] or len(set(stress['seeds'])) != len(stress['seeds']):
        raise ValueError('Seeds vazias ou repetidas')
    if len(stress['levels']) != 3 or 'original' in stress['levels']:
        raise ValueError('Defina tres intensidades, sem usar o nome original')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    levels = {'original': dict(drop_probability=0, box_noise_fraction=0, false_positives_per_frame=0),
              **stress['levels']}
    rows, inputs = [], {}
    for name in config['split']['validation']:
        sequence = Sequence.load(root, name, config['public_detector'])
        # Corrupt only the frozen, accepted detections; low scores cannot re-enter.
        source = sequence.public_detections()
        source = source[source[:, 5] >= config['score_threshold']]
        for level, parameters in levels.items():
            for seed in stress['seeds']:
                key = Path(level) / f'seed-{seed}' / name
                # Stable sequence-specific RNG, independent of loop order.
                rng_seed = int.from_bytes(hashlib.sha256(f'{name}/{seed}'.encode()).digest()[:8], 'little')
                detections = corrupt_detections(source, sequence.length, sequence.width,
                                               sequence.height, parameters, rng_seed)
                csv_path = output / 'detections' / key.with_suffix('.csv')
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(csv_path, detections, delimiter=',', fmt='%.17g')
                result, predicted = evaluate_sequence(sequence, detections, config, 'external')
                save_predictions(output / 'predictions' / 'baseline' / key.with_suffix('.txt'), predicted)
                common = {'level': level, 'seed': seed, 'parameters': parameters}
                rows.append({**result, **common, 'model': 'baseline'})
                inputs[key.as_posix()] = {'corrupted_sha256': digest(csv_path),
                    'source_sha256': digest(sequence.directory/'det/det.txt'),
                    'gt_sha256': digest(sequence.directory/'gt/gt.txt')}
                if temporal_predictions:
                    path = Path(temporal_predictions) / key.with_suffix('.txt')
                    predicted = load_predictions(path)
                    if np.any(predicted[:, 0] > sequence.length):
                        raise ValueError(f'Previsoes fora da sequencia: {path}')
                    raw = sequence.ground_truth()
                    mask = evaluation_mask(raw, predicted[:, 0], predicted[:, 2:6], config['ignore_iou'])
                    tracking = evaluate(valid_gt(raw), predicted[mask], config['evaluation_iou'])
                    count = tracking['gt_identities']
                    rows.append({**result, **tracking, **common, 'model': 'temporal',
                        'identity_ratio': tracking['predicted_identities']/count if count else None,
                        'switches_per_gt_identity': tracking['id_switches']/count if count else None,
                        'raw_predicted_identities': int(len(np.unique(predicted[:, 1]))),
                        'ignored_predictions': int((~mask).sum())})
                    inputs[key.as_posix()]['temporal_predictions_sha256'] = digest(path)
                print(f'{name} {level} seed={seed}: mAP={result["map_50_95"]:.4f} '
                      f'IDF1={result["idf1"]:.4f}', flush=True)
    payload = {'config': config, 'stress': stress, 'results': rows, 'inputs': inputs,
               'complete_part5': bool(temporal_predictions),
               'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                 ['parte5.py', 'parte1.py', 'mot_data.py', 'metrics.py', 'geometry.py',
                  'baseline.py', 'detection_metrics.py']},
               'numpy_version': np.__version__}
    (output/'results.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    flat = [{k: v for k, v in row.items() if k not in ('parameters', 'ap_by_iou')} for row in rows]
    with (output/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plot_results(rows, output/'estresse.png', list(levels))
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/MOT17')
    parser.add_argument('--output', default='outputs/parte5')
    parser.add_argument('--config', default='configs/parte1.json')
    parser.add_argument('--stress-config', default='configs/parte5.json')
    parser.add_argument('--temporal-predictions', help='MOT: <nivel>/seed-<seed>/<MOT17-XX>.txt')
    args = parser.parse_args()
    run(args.root, args.output, args.config, args.stress_config, args.temporal_predictions)
