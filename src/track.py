import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor


class Tracker:
    def __init__(self, device, model_id="facebook/sam2.1-hiera-tiny"):
        self.device = device
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(model_id).to(device)
        # streaming session so frames are fed one at a time
        self.session = self.processor.init_video_session(inference_device=device)
        self.started = False

    @torch.no_grad()
    def step(self, idx, frame, box=None):
        inputs = self.processor(images=frame, device=self.device, return_tensors="pt")
        if box is not None:
            self.processor.add_inputs_to_inference_session(
                inference_session=self.session,
                frame_idx=idx,
                obj_ids=1,
                input_boxes=[[[float(v) for v in box]]],
                original_size=inputs.original_sizes[0],
            )
            self.started = True
        if not self.started:
            return None, 0.0
        out = self.model(inference_session=self.session, frame=inputs.pixel_values[0], frame_idx=idx)
        mask = self.processor.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes)[0]
        score = torch.sigmoid(out.object_score_logits).item()
        return mask[0, 0].cpu().numpy(), score


def mask_to_box(mask):
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
