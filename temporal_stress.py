"""Aplica checkpoint fixo a todas as degradacoes da Parte 5, sem retreino."""
import argparse
import json
from pathlib import Path

import numpy as np

from mot_data import Sequence, load_config, save_predictions
from parte5 import run as prepare_stress
from temporal import load_motion, track_sequence


def run(root='data/MOT17', config_path='configs/parte1.json',
        stress_path='configs/parte5.json', checkpoint='checkpoints/temporal_mot17.pt',
        output='outputs/parte5', predictions='outputs/temporal_stress', device='auto', max_age=8):
    config = load_config(config_path)
    # Gerar entradas idênticas para ambos os rastreadores, inclusive controle.
    prepare_stress(root, output, config_path, stress_path)
    model, training = load_motion(checkpoint, device)
    stress = json.loads(Path(stress_path).read_text(encoding='utf-8'))
    for name in config['split']['validation']:
        sequence = Sequence.load(root, name, config['public_detector'])
        for level in ['original', *stress['levels']]:
            for seed in stress['seeds']:
                key = Path(level)/f'seed-{seed}'/name
                csv_path = Path(output)/'detections'/key.with_suffix('.csv')
                detections = np.loadtxt(csv_path, delimiter=',', ndmin=2) if csv_path.stat().st_size else np.empty((0, 6))
                pred = track_sequence(sequence, detections, model,
                                      config['score_threshold'], config['association_iou'], max_age)
                save_predictions(Path(predictions)/key.with_suffix('.txt'), pred)
                print(f'{name} {level} seed={seed}: {len(pred)} previsões', flush=True)
    return prepare_stress(root, output, config_path, stress_path, predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/MOT17')
    parser.add_argument('--config', default='configs/parte1.json')
    parser.add_argument('--stress-config', default='configs/parte5.json')
    parser.add_argument('--checkpoint', default='checkpoints/temporal_mot17.pt')
    parser.add_argument('--output', default='outputs/parte5')
    parser.add_argument('--predictions', default='outputs/temporal_stress')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--max-age', type=int, default=8)
    a = parser.parse_args()
    run(a.root, a.config, a.stress_config, a.checkpoint,
        a.output, a.predictions, a.device, a.max_age)
