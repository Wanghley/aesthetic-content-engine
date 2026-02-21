import os
import torch
import cv2
import rawpy
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import AutoProcessor, AutoModel, CLIPModel, CLIPProcessor
from pillow_heif import register_heif_opener

# Register HEIF support for PIL
register_heif_opener()

class LifeIndexer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Model 1: Aesthetic Scoring
        self.aes_model = AutoModel.from_pretrained("shadowlilac/aesthetic-shadow-v2-tiny-vit").to(self.device)
        self.aes_proc = AutoProcessor.from_pretrained("shadowlilac/aesthetic-shadow-v2-tiny-vit")
        
        # Model 2: CLIP for Semantic understanding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _load_media(self, path):
        """Universal loader for standard images, HEIC, RAW, and Video frames."""
        ext = os.path.splitext(path)[1].lower()
        try:
            # 1. Handle Video (.mov, .mp4, .avi)
            if ext in ('.mov', '.mp4', '.avi'):
                cap = cv2.VideoCapture(path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Capture frame at 50% to get a representative shot
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 2. Handle RAW formats (.dng, .arw, .cr2, .nef)
            elif ext in ('.dng', '.arw', '.cr2', '.nef'):
                with rawpy.imread(path) as raw:
                    rgb = raw.postprocess()
                    return Image.fromarray(rgb)

            # 3. Handle Standard & HEIC (supported via register_heif_opener)
            else:
                return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error decoding {path}: {e}")
            return None

    def process_folder(self, folder_path):
        data = []
        thumb_dir = "outputs/thumbnails"
        os.makedirs(thumb_dir, exist_ok=True)
        
        valid_exts = ('.png', '.jpg', '.jpeg', '.heic', '.mov', '.mp4', '.dng', '.arw', '.cr2', '.nef')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        
        for file in tqdm(files, desc="Indexing Memories"):
            path = os.path.join(folder_path, file)
            img = self._load_media(path)
            
            if img is None:
                continue

            # Save thumbnail for videos so the gallery can display them
            if file.lower().endswith(('.mov', '.mp4', '.avi')):
                img.save(os.path.join(thumb_dir, f"{file}.jpg"), "JPEG")

            try:
                # Get Aesthetic Score
                inputs = self.aes_proc(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    score = self.aes_model(**inputs).logits.item()
                
                # Get CLIP Embedding
                inputs = self.clip_proc(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    features = self.clip_model.get_image_features(**inputs)
                
                data.append({
                    "filename": file,
                    "aesthetic_score": score,
                    "embedding": features.cpu().numpy().tolist()[0],
                    "timestamp": os.path.getmtime(path)
                })
            except Exception as e:
                print(f"Skipping {file} due to model error: {e}")
        
        df = pd.DataFrame(data)
        df.to_pickle("outputs/manifest.pkl")
        return df