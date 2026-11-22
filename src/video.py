import cv2
import numpy as np


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def draw(frame, mask, box, lost=False):
    out = frame.copy()
    if mask is not None:
        out[mask > 0] = (0.5 * out[mask > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    if box is not None:
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    if lost:
        cv2.putText(out, "lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return out
