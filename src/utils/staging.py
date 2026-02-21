import shutil
import os

def stage_content(selected_filenames, post_name="final_update"):
    staging_dir = f"outputs/staging/{post_name}"
    os.makedirs(staging_dir, exist_ok=True)
    
    for i, filename in enumerate(selected_filenames):
        src = f"data/{filename}"
        # Rename to 1.jpg, 2.jpg for easy carousel ordering
        ext = os.path.splitext(filename)[1]
        dst = f"{staging_dir}/{i+1}{ext}"
        
        shutil.copy(src, dst)
        print(f"📸 Staged: {filename} -> {dst}")