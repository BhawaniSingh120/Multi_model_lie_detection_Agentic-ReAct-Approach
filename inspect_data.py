import os
import pandas as pd
from src import config

def inspect_own_dataset():
    path = os.path.join(config.DATA_DIR, "own_dataset", "Own Dataset", "Labelling.xlsx")
    if os.path.exists(path):
        print("--- Own Dataset Labelling.xlsx ---")
        try:
            df = pd.read_excel(path)
            print("Columns:", df.columns.tolist())
            if not df.empty:
                print(df.iloc[0]) 
            # Check for label column like 'Label', 'Lie', 'Truth'
        except Exception as e:
            print(f"Error reading xlsx: {e}")

def inspect_real_life():
    base = os.path.join(config.DATA_DIR, "real_life", "Real-life_Deception_Detection_2016")
    clips_dir = os.path.join(base, "Clips")
    print("\n--- Real Life Clips ---")
    if os.path.exists(clips_dir):
        print("Folders/Files in Clips:", os.listdir(clips_dir)[:10])
        
    anno_dir = os.path.join(base, "Annotation")
    if os.path.exists(anno_dir):
        print("\n--- Real Life Annotations ---")
        print("Files:", os.listdir(anno_dir))

if __name__ == "__main__":
    inspect_own_dataset()
    inspect_real_life()
