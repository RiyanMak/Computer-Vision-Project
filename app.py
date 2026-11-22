import os
import tempfile

import gradio as gr
import torch

from run import save, track
from src.detect import Detector
from src.embed import Embedder
from src.track import Tracker
from src.video import read_frames

device = "cuda" if torch.cuda.is_available() else "cpu"
detector = Detector(device)
tracker = Tracker(device)
embedder = Embedder(device)


def run(video, prompt, redetect):
    frames, fps = read_frames(video)
    rows, vis = track(frames, prompt, detector, tracker, embedder, redetect=redetect)
    name = os.path.splitext(os.path.basename(video))[0]
    return save(rows, vis, fps, tempfile.mkdtemp(), name)


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Video(label="video"),
        gr.Textbox(label="describe the object", placeholder="the red backpack"),
        gr.Checkbox(value=True, label="re-detect when lost"),
    ],
    outputs=gr.Video(label="tracked"),
    title="Text-Guided Object Tracking",
)

if __name__ == "__main__":
    demo.launch()
