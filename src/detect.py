import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class Detector:
    def __init__(self, device, model_id="IDEA-Research/grounding-dino-tiny"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    @torch.no_grad()
    def __call__(self, frame, prompt, box_thresh=0.35, text_thresh=0.25):
        image = Image.fromarray(frame)
        # dino expects lowercase text ending in a period
        text = prompt.lower().strip().rstrip(".") + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        order = np.argsort(-scores)
        return boxes[order], scores[order]
