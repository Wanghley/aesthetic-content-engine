import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

class AestheticScorer:
    def __init__(self):
        # We use a model trained on the AVA dataset (Aesthetic Visual Analysis)
        self.model = AutoModel.from_pretrained("shadowlilac/aesthetic-shadow-v2-tiny-vit")
        self.processor = AutoProcessor.from_pretrained("shadowlilac/aesthetic-shadow-v2-tiny-vit")

    def get_score(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # This returns a logit that we can normalize to a 1-10 scale
        return outputs.logits.item()