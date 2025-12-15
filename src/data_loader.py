import os
import zipfile
import pandas as pd
import glob
from . import config

def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print(f"Zip file not found: {zip_path}")
        return
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def extract_bz2(bz2_path, extract_to):
    import bz2
    import shutil
    
    if not os.path.exists(bz2_path):
        print(f"bz2 file not found: {bz2_path}")
        return

    with bz2.BZ2File(bz2_path) as fr, open(extract_to, 'wb') as fw:
        shutil.copyfileobj(fr, fw)
    print(f"Extracted {bz2_path} to {extract_to}")

def load_metadata():
    records = []

    # 1. Real-Life Dataset
    real_life_base = os.path.join(config.DATA_DIR, "real_life", "Real-life_Deception_Detection_2016", "Clips")
    if os.path.exists(real_life_base):
        for label_name in ["Deceptive", "Truthful"]:
            folder_path = os.path.join(real_life_base, label_name)
            if not os.path.exists(folder_path):
                continue
            
            label = 1 if label_name == "Deceptive" else 0
            # Files are likely mp4
            files = glob.glob(os.path.join(folder_path, "*.mp4"))
            for f in files:
                records.append({
                    "video_path": f,
                    "label": label,
                    "dataset": "real_life"
                })
    
    # 2. Own Dataset
    own_base = os.path.join(config.DATA_DIR, "own_dataset", "Own Dataset")
    xlsx_path = os.path.join(own_base, "Labelling.xlsx")
    
    if os.path.exists(xlsx_path):
        try:
            df = pd.read_excel(xlsx_path)
            # Columns: Video Name, Label (Truthful/Lie?)
            # Map Label
            for _, row in df.iterrows():
                video_name = row['Video Name']
                label_str = str(row['Label']).strip().lower()
                
                # Check extension
                # Try finding file with .MOV or .mp4
                video_file = os.path.join(own_base, f"{video_name}.MOV")
                if not os.path.exists(video_file):
                    video_file = os.path.join(own_base, f"{video_name}.mp4")
                    if not os.path.exists(video_file):
                        # Try exact match if extension was included (unlikely based on head)
                        continue
                
                label = 0 # Default Truthful
                if "lie" in label_str or "deceptive" in label_str:
                    label = 1
                elif "truth" in label_str:
                    label = 0
                
                records.append({
                    "video_path": video_file,
                    "label": label,
                    "dataset": "own_dataset"
                })
        except Exception as e:
            print(f"Error processing Own Dataset excel: {e}")

    # Create DataFrame
    metadata_df = pd.DataFrame(records)
    print(f"Found {len(metadata_df)} videos.")
    print(metadata_df['label'].value_counts())
    
    out_path = os.path.join(config.DATA_DIR, "metadata.csv")
    metadata_df.to_csv(out_path, index=False)
    print(f"Saved metadata to {out_path}")

def prepare_data():
    # Only extract if dirs don't exist
    if not os.path.exists(os.path.join(config.DATA_DIR, "real_life")):
        extract_zip(config.REAL_LIFE_ZIP, os.path.join(config.DATA_DIR, "real_life"))
    
    if not os.path.exists(os.path.join(config.DATA_DIR, "own_dataset")):
        extract_zip(config.OWN_DATASET_ZIP, os.path.join(config.DATA_DIR, "own_dataset"))
    
    if not os.path.exists(config.PREDICTOR_MODEL):
         extract_bz2(config.SHAPE_PREDICTOR_PATH, config.PREDICTOR_MODEL)
         
    load_metadata()

if __name__ == "__main__":
    prepare_data()
