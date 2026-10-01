"""Associação própria usando a última caixa observada e Hungarian."""
import numpy as np
from geometry import match_iou


class IoUTracker:
    def __init__(self, threshold=0.3, max_age=2):
        self.threshold, self.max_age = threshold, max_age
        self.tracks = {}
        self.next_id = 1
        self.last_frame = 0

    def update(self, frame, boxes):
        if frame <= self.last_frame:
            raise ValueError('Os quadros devem ser estritamente crescentes')
        self.last_frame = frame
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        # Permite max_age quadros intermediários sem observação.
        self.tracks = {i: t for i, t in self.tracks.items() if frame-t[1] <= self.max_age+1}
        ids = list(self.tracks)
        pairs = match_iou([self.tracks[i][0] for i in ids], boxes, self.threshold)
        assignment = {j: ids[i] for i, j in pairs}
        output = []
        for j, box in enumerate(boxes):
            if j not in assignment:
                assignment[j] = self.next_id
                self.next_id += 1
            identity = assignment[j]
            self.tracks[identity] = (box.copy(), frame)
            output.append([frame, identity, *box])
        # Tracks sem observação ficam na memória, mas não geram caixas fictícias.
        return output
