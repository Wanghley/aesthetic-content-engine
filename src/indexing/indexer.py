import os
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import AutoProcessor, AutoModel, CLIPModel, CLIPProcessor

class LifeIndexer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Model 1: Aesthetic Scoring
        self.aes_model = AutoModel.from_pretrained("shadowlilac/aesthetic-shadow-v2-tiny-vit").to(self.device)
        self.aes_proc = AutoProcessor.from_pretrained("shadowlilac/aesthetic-shadow-v2-tiny-vit")
        
        # Model 2: CLIP for Semantic understanding (Grouping your year)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def process_folder(self, folder_path):
        data = []
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for file in tqdm(files, desc="Indexing Memories"):
            path = os.path.join(folder_path, file)
            try:
                img = Image.open(path).convert("RGB")
                
                # Get Aesthetic Score
                inputs = self.aes_proc(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    score = self.aes_model(**inputs).logits.item()
                
                # Get CLIP Embedding (for clustering later)
                inputs = self.clip_proc(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    features = self.clip_model.get_image_features(**inputs)
                
                data.append({
                    "filename": file,
                    "aesthetic_score": score,
                    "embedding": features.cpu().numpy().tolist()[0],
                    "timestamp": os.path.getmtime(path) # Fallback if EXIF is missing
                })
            except Exception as e:
                print(f"Skipping {file}: {e}")
        
        df = pd.DataFrame(data)
        df.to_pickle("outputs/manifest.pkl") # Pickle preserves the embedding lists
        return df