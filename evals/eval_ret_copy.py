import numpy as np
from torch import nn
import torch
import yaml
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets.coco import CocoCaptions
import json
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('/user/home/a.simplicio/Requeijao')
from fromage import models
from fromage import utils
from sklearn.metrics import recall_score

#model_dir = '/user/home/a.simplicio/Fromage/runs/Gloria'
#model_dir = '/user/home/a.simplicio/Requeijao/runs/gloria_deepL_resume'
model_dir = '/user/home/a.simplicio/Requeijao/runs/Gervasio_1'
model = models.load_fromage(model_dir)
feature_extractor = model.model.feature_extractor

print(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"


valid_root =  '/storagebk/datasets/ms-coco/val2017'
#valid_captions = '/user/home/a.simplicio/Requeijao/evals/captions_val_deepl1.json'
#valid_captions = '/user/home/a.simplicio/Requeijao/evals/captions_val_portugues2017.json'
valid_captions = '/user/home/a.simplicio/Requeijao/evals/visdial_val_portuguese.json'
img_dir = '/storagebk/datasets/ms-coco/val2017'

def preprocess(images):
    """
    Custom collate function to convert PIL Images to tensors.
    """
    #images, texts = zip(*batch)
    
    # Convert images to tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Add any other transformations you need
    ])
    
    
    return transform(images)


def get_pixel_values(img):
    pixel_values = utils.get_pixel_values_for_model(feature_extractor, img)
    if torch.cuda.is_available():
        pixel_values = pixel_values.bfloat16()
        pixel_values = pixel_values.cuda()
    return pixel_values


def compute_similarity(image_features, text_features, bs = 1000):
    # compute similarity
    max_pairs = image_features.shape[0]
    similarity_scores = torch.zeros(max_pairs, max_pairs)
    for v in range(0, max_pairs, bs):
        for t in range(0, max_pairs, bs):
            print('Processing Visual '+str(v)+' Text '+str(t), end='\r')
            batch_visual_emb = image_features[v:v+bs]
            batch_caption_emb = text_features[t:t+bs]
            logits = batch_visual_emb @ batch_caption_emb.t()
            similarity_scores[v:v+bs,t:t+bs] = logits

    print('Done similarity')
    print(similarity_scores.shape)
    print(similarity_scores)
    return similarity_scores

def compute_retrieval(a2b_sims, return_ranks=True):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # loop source embedding indices
    for index in range(npts):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    report_dict = {"r1": r1, "r5": r5, "r10": r10, "r50": r50, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10}

    if return_ranks:
        return report_dict, (ranks, top1)
    else:
        return report_dict


#with open(valid_captions, 'r', encoding='utf-8') as f:
#    data = json.load(f)
#    
#annotations  = data['annotations']
 
# fwd all samples
valid_dataset = CocoCaptions(root = valid_root,
                        annFile = valid_captions, transform = get_pixel_values)
valid_dataloader = DataLoader(valid_dataset, batch_size = 1)


single_caption = True
#for example_idx in tqdm(range(len(annotations))):

image_features = []
text_features = []
for batch_idx, batch in enumerate(tqdm(valid_dataloader)):
    #print('Evaluating batch {}/{}'.format(batch_idx, len(valid_dataloader)), end = "\r")
    images, texts = batch

    if single_caption:
        texts = [texts[0][0]+ '[RET]'] 
    else:
        texts = [txt[0]+ '[RET]' for txt in texts]
    #print(0)

    input_ids = model.model.tokenizer(texts, add_special_tokens=True, return_tensors="pt", padding=True).input_ids
    input_ids = input_ids.cuda()
    input_embs = model.model.input_embeddings(input_ids)  # (N, T, D)
    generated_ids, output_embs, _ = model(input_embs, None, None, generate=True, num_words=1, temperature=0.0)
    embeddings = output_embs[0]
    #print(1)
    full_input_ids = torch.cat([input_ids, generated_ids], dim=1)
    text_emb = embeddings[:, -1, :] #ret emb
    if not single_caption:
        text_emb = text_emb.unsqueeze(0)
    #print(2)
   # print(images.get_device())
    #print(text_emb.get_device())
    image_emb =  model.model.get_visual_embs(images, mode='retrieval').cuda() #embed with image encoder
    #print(3)

    text_features.append(text_emb.detach().cpu())
    image_features.append(image_emb.detach().cpu())

    #print(4)
text_features = [tensor.float() for tensor in text_features]

image_features = [tensor.float() for tensor in image_features]


image_features = np.concatenate(image_features , axis=0)[:, 0, :]
text_features  = np.concatenate( text_features, axis=0)
image_features = torch.tensor(image_features)
text_features = torch.tensor(text_features)
#image_features = torch.cat(all_visual_embs, 0)
#text_features = torch.cat(all_text_embs, 0)
print('Done forward')

 #normalized features
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

if not single_caption:
    for cap_idx in range(text_features.shape[1]):
        similarity_scores = compute_similarity(image_features, text_features[:,cap_idx,:])
        i2t_dict = compute_retrieval(similarity_scores.numpy())
        t2i_dict = compute_retrieval(similarity_scores.t().numpy())
        print(cap_idx, 'i2t', i2t_dict)
        print(cap_idx, 't2i', t2i_dict)
else:
    similarity_scores = compute_similarity(image_features, text_features)
    i2t_dict = compute_retrieval(similarity_scores.numpy())
    t2i_dict = compute_retrieval(similarity_scores.t().numpy())
    print('i2t', i2t_dict)
    print('t2i', t2i_dict)
