import os
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        if 'content' in data.columns and 'score' in data.columns:
            data = data[['content', 'score']]
            data.columns = ['text', 'label']
            data['label'] = data['label'] - 1
            return data
        else:
            print("❌ Colonnes manquantes.")
            return None
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None
