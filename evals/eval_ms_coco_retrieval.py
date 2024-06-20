
import numpy as np
import collections
import copy
import json
import os
import torch
from transformers import logging
from tqdm import tqdm
logging.set_verbosity_error()

from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('/user/home/a.simplicio/Requeijao')
from fromage import models
from fromage import utils


def get_pixel_values_from_path(path: str, feature_extractor):
    """Helper function for getting images pixels from a local path."""
    img = Image.open(path)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    pixel_values = utils.get_pixel_values_for_model(feature_extractor, img)
    if torch.cuda.is_available():
        pixel_values = pixel_values.bfloat16()
        pixel_values = pixel_values.cuda()
    return pixel_values[None, ...]



if __name__ == "__main__":
    # Load model used in the paper.
    model_dir = '/user/home/a.simplicio/Requeijao/runs/gloria_deepL_resume' #'./fromage_model/'

    model = models.load_fromage(model_dir)


    base_dir = '/user/home/a.simplicio/Requeijao/evals'
    split = 'val'
    img_dir = '/storagebk/datasets/ms-coco/val2017'

    #with open(os.path.join(base_dir, 'captions_val_portugues2017_0.json'), 'r', encoding='utf-8') as f:
    with open('/user/home/a.simplicio/Requeijao/evals/captions_val_deepl1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    annotations  = data['annotations']
 
        


    # Then, we compute the image features and text features for each VisDial example:
    topk = (1, 5, 10)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none').cuda()

    all_visual_embs = []
    all_text_embs = []

    for example_idx in tqdm(range(len(annotations))):
        ann = annotations[example_idx]
        image_id = str(ann['image_id']).rjust(12, '0')
        caption = ann['caption'] + '[RET]'

        with torch.no_grad():
            images = get_pixel_values_from_path(
                os.path.join(img_dir, f'{image_id}.jpg'),
                model.model.feature_extractor)
            visual_embs = model.model.get_visual_embs(images, mode='retrieval')
            
            input_ids = model.model.tokenizer(caption, add_special_tokens=True, return_tensors="pt").input_ids
            input_ids = input_ids.cuda()
            input_embs = model.model.input_embeddings(input_ids)  # (N, T, D)
            generated_ids, output_embs, _ = model(input_embs, None, None, generate=True, num_words=1, temperature=0.0)
            embeddings = output_embs[0]

            full_input_ids = torch.cat([input_ids, generated_ids], dim=1)
            ret_emb = embeddings[:, -1, :]  

            all_visual_embs.append(visual_embs.cpu().detach().float().numpy())
            all_text_embs.append(ret_emb.cpu().detach().float().numpy())

    # Compute scores over the whole dataset:
    scores = np.concatenate(all_visual_embs, axis=0)[:, 0, :] @ np.concatenate(all_text_embs, axis=0).T
    scores = torch.tensor(scores).float()
    #assert scores.shape == (2064, 2064), scores.shape


    # Finally, we can compute the Recall@k scores:
    _, preds = scores.topk(max(topk))
    print(preds [:5])
    for k in topk:
        labels = torch.arange(preds.shape[0])
        correct = torch.any(preds[:, :k] == labels[:, None], axis=1).sum()
        acc = correct / preds.shape[0]
        print(f'top-k, k={k}, acc={acc:.5f}')
    print('=' * 20)
    print(labels)
    print(correct)



