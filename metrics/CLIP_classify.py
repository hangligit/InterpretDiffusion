from PIL import Image
import requests
import os, glob
import pandas as pd
import numpy as np
import re
from transformers import CLIPProcessor, CLIPModel
import argparse
import torch
# taken from https://github.com/rohitgandikota/erasing/tree/main/eval-scripts

def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def add_winobias_metrics(df):
    def deviation_ratio(list_of_counts):
        r = np.array(list_of_counts)
        r=r/np.sum(r)
        ref=np.ones((len(r)))/len(r)
        return np.abs(r-ref).max()/(1-1/len(r))
    columns = df.columns
    df['deviation_ratio'] = df.apply(lambda row: deviation_ratio(row[columns]), axis=1)
    return df


def CLIP_classification_function(im_dir, attributes, model, processor, from_case=0, till_case=1000000, return_df=False, args=None):
    images = os.listdir(im_dir)
    images = [im for im in images if '.png' in im or '.jpg' in im]
    images = sorted_nicely(images)
    ratios = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for image in images:
        try:
            case_number = int(image.split('_')[0].replace('.png','').replace('.jpg',''))
            if case_number < from_case or case_number > till_case:
                continue

            im = Image.open(os.path.join(im_dir, image))
            inputs = processor(text=attributes, images=im, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1).cpu()  # we can take the softmax to get the label probabilities
            tmax = probs.max(1, keepdim=True)[0]
            mask = list(probs.ge(tmax)[0].float().numpy())
            ratios[case_number] = ratios.get(case_number, []) + [mask]
        except Exception:
            ratios[case_number] = ratios.get(case_number, []) + [[0]*len(attributes)]

    columns = [f"{att.replace(' ','_')}_bias_cnt" for att in attributes]
    df=pd.DataFrame()
    for col in columns:
        df[col] = np.nan
    for key in ratios.keys():
        for idx, col in enumerate(columns):
            df.loc[key,col] = np.mean(np.array(ratios[key])[:,idx])
    if return_df:
        return df
    if not len(columns)==2: print("WARNING - CLIP_classification_function: #attributes is not 2")
    return df.mean().tolist()[0]


def evaluate_winobias(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    clip_attributes = args.clip_attributes
    logging = []
    
    im_dir = args.im_dir
    professions = os.listdir(im_dir)
    professions = sorted([prof for prof in professions if not '.csv' in prof])
    
    for profession in professions:  
        im_dir = os.path.join(args.im_dir, profession)
        df = CLIP_classification_function(im_dir=im_dir, attributes=clip_attributes, model=clip_model, processor=processor, return_df=True)
        result = {'profession': profession}
        sums = df.sum().to_dict()
        result.update(sums)
        logging.append(result)
        print(result)
        
    logging = pd.DataFrame(logging)
    logging = add_winobias_metrics(logging.set_index('profession'))
    logging.loc['mean'] = logging.mean()
    
    save_name = '_'.join([s.replace(' ', '_') for s in args.clip_attributes])
    save_name += '_result.csv'
    save_path = os.path.join(args.im_dir, save_name)
    logging.to_csv(save_path, index=True)
    print(f'CLIP classification results saved to {save_path}')
    print(f'Mean CLIP classification results: {logging.loc["mean"].to_dict()}')


def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    
    clip_attributes = args.clip_attributes
    im_dir = args.im_dir
    df = CLIP_classification_function(im_dir=im_dir, attributes=clip_attributes, model=clip_model, processor=processor, return_df=True)

    sums = df.sum().to_dict()
    df = add_winobias_metrics(pd.DataFrame([sums]))
    
    save_name = '_'.join([s.replace(' ', '_') for s in args.clip_attributes])
    save_name += '_result.csv'
    save_path = os.path.join(args.im_dir, save_name)
    df.to_csv(save_path, index=True)
    print(df)
    print(f'CLIP classification results saved to {save_path}')


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'CLIP classification',
                    description = 'Takes the path to images and gives CLIP classification scores')
    parser.add_argument('--im_dir', help='dir to folders of winobias images', type=str, required=True)
    parser.add_argument('--clip_attributes', type=str, required=True, nargs='+')
    parser.add_argument('--eval_type', type=str, default=None, choices=['winobias'])
    
    args = parser.parse_args()
    
    if args.eval_type=='winobias':
        evaluate_winobias(args)
    else:
        evaluate(args)
