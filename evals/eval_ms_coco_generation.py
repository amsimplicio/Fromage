import os
import torch
from tqdm import tqdm
import json
import sys
from PIL import Image
sys.path.append('/user/home/a.simplicio/Requeijao')
from fromage import models
from fromage import utils
import argparse



def generate_captions( model, root_path):
    results = []
    for file in tqdm(os.listdir(root_path)):
        img_id = int(file.split(".")[0])
        img_path = os.path.join(root_path, file)
       # image = utils.get_image_from_path(img_path)
        image = Image.open( img_path)
        image = image.resize((224, 224))
        pixel_values = utils.get_pixel_values_for_model(model.model.feature_extractor, image)
        pixel_values = pixel_values.to(device=model.model.logit_scale.device, dtype=model.model.logit_scale.dtype)
        pixel_values = pixel_values[None, ...]
        imginp = model.model.get_visual_embs(pixel_values, mode='captioning')

        generated_ids, _, _ = model(
                                 imginp, None, None, generate=True, temperature=0.0, top_p=1.0)
        

        predicted_answer = model.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predicted_answer = utils.truncate_caption(predicted_answer).strip()
        results.append({
            "image_id": img_id,
            "caption": predicted_answer
        })
    return results



def generate(path, filename):
# Load model used in the paper.


    model = models.load_fromage(path)

    #load questions
    #annFile     ='/user/home/a.simplicio/Fromage/evals/captions_val_portugues2017.json'
    #with open(annFile, 'r') as f:
    #    data = json.load(f)
    #anns  = data['annotations']
    results = generate_captions(model, "/storagebk/datasets/ms-coco/val2017")

    with open(filename, "w", encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False)


#generate("/user/home/a.simplicio/Fromage/runs/GPTNeo", 'mscoco_captions_results_gptneo.json')

#generate("/user/home/a.simplicio/Requeijao/runs/Gervasio_1", 'mscoco_captions_results_gervasio_madlad.json')
generate("/user/home/a.simplicio/Requeijao/runs/Gervasio_8", 'mscoco_captions_results_gervasio_deepl.json')