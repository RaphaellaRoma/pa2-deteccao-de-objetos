"""Inferência de uma sequência MOT17 a partir de checkpoint, com vídeo MP4."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mot_data import Sequence, save_predictions
from temporal import load_motion, track_sequence
from visualization import identity_color


def render(sequence_dir, checkpoint='checkpoints/temporal_mot17.pt',
           output='outputs/inferencia/resultado.mp4', device='auto',
           score_threshold=0.5, association_iou=0.3, max_age=8, max_frames=None):
    directory = Path(sequence_dir)
    if not directory.is_dir() or '-' not in directory.name:
        raise FileNotFoundError(f'Diretório de sequência MOT17 ausente: {directory}')
    name, detector = directory.name.rsplit('-', 1)
    sequence = Sequence.load(directory.parent.parent, name, detector)
    model, training = load_motion(checkpoint, device)
    detections = sequence.public_detections()
    if max_frames is not None:
        detections = detections[detections[:, 0] <= max_frames]
    # Manter duração completa na entrega; max_frames é somente para ensaio local.
    if max_frames is None:
        pred = track_sequence(sequence, detections, model, score_threshold,
                              association_iou, max_age)
        length = sequence.length
    else:
        from temporal import TemporalTracker
        tracker = TemporalTracker(model, sequence.width, sequence.height, association_iou, max_age)
        rows = []
        length = min(max_frames, sequence.length)
        for f in range(1, length + 1):
            boxes = detections[(detections[:, 0] == f) &
                               (detections[:, 5] >= score_threshold), 1:5]
            rows.extend(tracker.update(f, boxes))
        pred = np.asarray(rows, float).reshape(-1, 6)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(output.with_suffix('.txt'), pred)
    width = min(sequence.width, 960)
    height = round(sequence.height * width / sequence.width)
    # x264 expects dimensions divisible by 16.
    width = max(16, width // 16 * 16)
    height = max(16, height // 16 * 16)
    sx, sy = width / sequence.width, height / sequence.height
    seen = set()
    with imageio.get_writer(output, fps=sequence.fps, codec='libx264', quality=7) as video:
        for frame in range(1, length + 1):
            with Image.open(sequence.image_path(frame)) as original:
                image = original.convert('RGB').resize((width, height))
            draw = ImageDraw.Draw(image)
            for _, identity, x, y, w, h in pred[pred[:, 0] == frame]:
                identity = int(identity)
                seen.add(identity)
                rgb = tuple(round(channel * 255) for channel in identity_color(identity))
                box = (round(x * sx), round(y * sy), round((x + w) * sx), round((y + h) * sy))
                draw.rectangle(box, outline=rgb, width=2)
                draw.text((box[0], max(0, box[1] - 12)), f'ID {identity}', fill=rgb)
            draw.rectangle((0, 0, 330, 25), fill=(0, 0, 0))
            draw.text((5, 5), f'{name}  quadro {frame}/{length}  IDs únicos: {len(seen)}',
                      fill=(255, 255, 255))
            video.append_data(np.asarray(image))
    return {'video': str(output), 'predictions': str(output.with_suffix('.txt')),
            'frames': length, 'unique_identities': len(seen), 'checkpoint_training': training}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence_dir', help='Ex.: data/MOT17/train/MOT17-09-FRCNN')
    parser.add_argument('--checkpoint', default='checkpoints/temporal_mot17.pt')
    parser.add_argument('--output', default='outputs/inferencia/resultado.mp4')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--max-frames', type=int, help='Somente ensaio rápido')
    args = parser.parse_args()
    print(render(args.sequence_dir, args.checkpoint, args.output, args.device,
                 max_frames=args.max_frames))
