import os

from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("Set the HF_TOKEN environment variable before running this script.")

api = HfApi(token=token)
api.upload_folder(
    folder_path="/home/khoina/LMUData/our_dataset",
    repo_id="KoiiVN/our_dataset",
    repo_type="dataset",
)

# Firstly, access the huggingface dataset website for get the file link.
# For instance: https://huggingface.co/datasets/KoiiVN/our_dataset/blob/main/our_dataset.tsv

# Secondly, running: `md5sum /home/khoina/LMUData/our_dataset/our_dataset.tsv` in the terminal 
# For output example: fcd85b662f9f4a9fe36ec917d8fffcac  /home/khoina/LMUData/our_dataset/our_dataset.tsv
# And copy md5 hash ("fcd85b662f9f4a9fe36ec917d8fffcac") for later using.
