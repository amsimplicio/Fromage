"""Extract image embeddings for a list of image urls.

Example usage:
    python extract_img_embs.py
"""
import torch
import sys
sys.path.append('/user/home/a.simplicio/Requeijao')
from fromage import models, utils

from PIL import Image
import os
import requests
from io import BytesIO
import pickle as pkl
import pandas as pd
from tqdm import tqdm
import zlib



def extract_embeddings_for_urls(data, emb_output_path: str, device: str = "cuda"):
    # Load model checkpoint.
    model = models.load_fromage("/user/home/a.simplicio/Requeijao/runs/Gervasio_8")
    model.eval()

    visual_encoder = "openai/clip-vit-large-patch14"
    feature_extractor = utils.get_feature_extractor_for_model(
        visual_encoder, train=False
    )

    output_data = {"paths": [], "embeddings": []}
    with torch.no_grad():
        for _, row in tqdm(data.iterrows()):
            try:
                img = Image.open( f"/storagebk/datasets/cc3m/training/{row['image']}")

                img_tensor = utils.get_pixel_values_for_model(feature_extractor, img)
                img_tensor = img_tensor[None, ...].to(device).bfloat16()
                img_emb = model.model.get_visual_embs(img_tensor, mode="retrieval")
                img_emb = img_emb[0, 0, :].cpu()
                output_data["paths"].append(row['url'])
                output_data["embeddings"].append(img_emb)
            except:
                pass

    with open(emb_output_path, "wb") as f:
        pkl.dump(output_data, f)


if __name__ == "__main__":



    data = pd.read_csv('/user/home/a.simplicio/cc3m_url.tsv', sep= '\t')

    # TODO: Replace with image urls
    
    #if image_urls == []:
    #    raise ValueError("Please replace `image_urls` with a list of image urls.")

    extract_embeddings_for_urls(data, "/user/home/a.simplicio/Requeijao/runs/Gervasio_8/cc3m_embeddings.pkl")