"""Treino e avaliacao da GRU de movimento nas sequencias MOT17."""
import argparse
import json
from pathlib import Path

import numpy as np

from metrics import evaluate
from mot_data import Sequence, evaluation_mask, load_config, save_predictions, valid_gt
from parte1 import evaluate_sequence
from temporal import load_motion, track_sequence, train_motion


def compare(root='data/MOT17', config_path='configs/parte1.json',
            checkpoint='checkpoints/temporal_mot17.pt', output='outputs/temporal',
            device='auto', max_age=8):
    config = load_config(config_path)
    model, training = load_motion(checkpoint, device)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name in config['split']['validation']:
        sequence = Sequence.load(root, name, config['public_detector'])
        detections = sequence.public_detections()
        baseline, baseline_pred = evaluate_sequence(sequence, detections, config, 'public')
        temporal_pred = track_sequence(sequence, detections, model,
                                       config['score_threshold'], config['association_iou'], max_age)
        raw = sequence.ground_truth()
        mask = evaluation_mask(raw, temporal_pred[:, 0], temporal_pred[:, 2:6], config['ignore_iou'])
        measured = evaluate(valid_gt(raw), temporal_pred[mask], config['evaluation_iou'])
        save_predictions(out/'predictions'/'baseline'/f'{name}.txt', baseline_pred)
        save_predictions(out/'predictions'/'temporal'/f'{name}.txt', temporal_pred)
        row = {'sequence': name, 'frames': sequence.length,
               'baseline': {k: baseline[k] for k in ('idf1', 'id_switches', 'fragmentations',
                   'predicted_identities', 'gt_identities', 'ap50', 'map_50_95')},
               'temporal': measured}
        results.append(row)
        print(f'{name}: IDF1 baseline={baseline["idf1"]:.4f} '
              f'temporal={measured["idf1"]:.4f}; IDs '
              f'{baseline["predicted_identities"]}/{measured["predicted_identities"]}', flush=True)
    payload = {'protocol': config['protocol'], 'detector': config['public_detector'],
               'score_threshold': config['score_threshold'],
               'association_iou': config['association_iou'], 'baseline_max_age': config['max_age'],
               'temporal_max_age': max_age, 'training': training, 'results': results}
    (out/'results.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate', 'all'])
    parser.add_argument('--root', default='data/MOT17')
    parser.add_argument('--config', default='configs/parte1.json')
    parser.add_argument('--checkpoint', default='checkpoints/temporal_mot17.pt')
    parser.add_argument('--output', default='outputs/temporal')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-windows', type=int, help='Amostragem apenas para ensaios rápidos')
    parser.add_argument('--max-age', type=int, default=8)
    parser.add_argument('--cell', choices=['GRU', 'RNN'], default='GRU',
                        help='Célula para treino; a GRU é o modelo temporal final')
    args = parser.parse_args(argv)
    if min(args.epochs, args.steps, args.batch_size) <= 0 or args.max_age < 0:
        parser.error('epochs, steps e batch-size devem ser positivos; max-age não negativo')
    config = load_config(args.config)
    if args.command in ('train', 'all'):
        train_motion(args.root, config, args.checkpoint, args.epochs, args.steps,
                     args.batch_size, args.seed, args.device, args.max_windows, args.cell)
    if args.command in ('evaluate', 'all'):
        compare(args.root, args.config, args.checkpoint, args.output, args.device, args.max_age)


if __name__ == '__main__':
    main()
