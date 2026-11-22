import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class Embedder:
    def __init__(self, device, model_id="openai/clip-vit-base-patch32"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device)

    @torch.no_grad()
    def __call__(self, frame, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = Image.fromarray(frame[y1:y2, x1:x2])
        inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
        feat = self.model.get_image_features(**inputs)
        if not torch.is_tensor(feat):
            feat = feat.pooler_output
        return torch.nn.functional.normalize(feat, dim=-1)[0].cpu()
