import pandas as pd
import base64
from PIL import Image
import io
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

df = pd.read_csv('/home/khoina/LMUData/NarrativeInfoVQA_Test/val_full_data.tsv', sep='\t')

def encode_row(row):
    img_path, question, answer, idx = row

    try:
        with Image.open(img_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"Error image {img_path}: {e}")
        img_base64 = ""

    return {
        'index': idx,
        'question': question,
        'answer': answer,
        'image': img_base64
    }

rows = [(df.iloc[i]["image_path"], 
         df.iloc[i]["question"],
         df.iloc[i]["answer"],
         i) for i in range(len(df))]

with Pool(cpu_count()) as p:
    results = list(tqdm(p.imap(encode_row, rows), total=len(rows)))

df_out = pd.DataFrame(results)
df_out.to_csv('/home/khoina/LMUData/our_dataset.tsv', sep='\t', index=False)