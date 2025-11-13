import pandas as pd
import base64
from PIL import Image
import io
from tqdm import tqdm

df = pd.read_csv('~/LMUData/our_raw_dataset.tsv', sep='\t')

def encode_image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

data = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row['image_path']
    question = row['question']
    answer = row['answer']
    
    data.append({
        'index': i,
        'question': question,
        'answer': answer,
        'image': encode_image_to_base64(img_path),
    })

df = pd.DataFrame(data)
df.to_csv('~/LMUData/our_dataset/our_dataset.tsv', sep='\t', index=False)
