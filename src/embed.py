import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def make_head(dim=512, out=256):
    return torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, out))


class Embedder:
    def __init__(self, device, model_id="openai/clip-vit-base-patch32", head=None):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.head = None
        if head:
            # basketball head
            self.head = make_head().to(device)
            self.head.load_state_dict(torch.load(head, map_location=device))
            self.head.eval()

    @torch.no_grad()
    def features(self, frame, boxes):
        crops = [Image.fromarray(frame[int(y1):int(y2), int(x1):int(x2)]) for x1, y1, x2, y2 in boxes]
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        feat = self.model.get_image_features(**inputs)
        if not torch.is_tensor(feat):
            feat = feat.pooler_output
        return feat

    @torch.no_grad()
    def __call__(self, frame, box):
        feat = self.features(frame, [box])
        if self.head is not None:
            feat = self.head(feat)
        return torch.nn.functional.normalize(feat, dim=-1)[0].cpu()
