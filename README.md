# Text-Guided Object Tracking in Real-World Videos

Track an object in a video from a text description using Grounding DINO and SAM 2.

## Setup

```
pip install -r requirements.txt
```

Runs on CPU but a GPU is much faster.

## Run

```
python run.py --video clip.mp4 --prompt "the red backpack"
```

Writes `outputs/<clip>_tracked.mp4` and `outputs/<clip>_boxes.csv` (box, SAM 2 object score, lost and reinit flags per frame).

Add `--no-redetect` for the baseline without re-detection.
