import glob
import os
import json
from PIL import Image
import numpy as np
import torch


def int_to_onehot(x, n):
    if not isinstance(x, list):
        x = [x]
    assert isinstance(x[0], int)
    x = torch.tensor(x).long()
    v = torch.zeros(n)
    v[x] = 1.
    return v


random_select = lambda l: l[np.random.choice(len(l))]
top_select = lambda l: l[0]


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform, tokenizer, max_concept_length, select):
        image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")), key=lambda x:int(x.split('/')[-1].split('.')[0]))
        self.image_paths = image_paths
        self.transform = transform
        self.tokenizer=tokenizer
        self.concept_dict=json.load(open(image_folder+'/concept_dict.json','r'))
        self.max_concept_length=max_concept_length
        if select=="top":
            self.select_method = top_select
        elif select=="random":
            self.select_method = random_select
        else:
            raise NotImplementedError(self.select_method)
        self.labels = json.load(open(image_folder+'/labels.json','r'))

    def __getitem__(self, index):
        input_prompt, target_concept = self.select_method(self.labels[index])
        input_prompt=self.tokenizer([input_prompt])[0]
        target_concept = [self.concept_dict[c] for c in target_concept]
        target_concept = int_to_onehot(target_concept, self.max_concept_length)
        image_path = self.image_paths[index]
        x = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, input_prompt, target_concept

    def __len__(self):
        return len(self.image_paths)


def get_dataloader(image_folder, batch_size, transform, tokenizer, collate_fn=None, num_workers=4, shuffle=False, max_concept_length=100, select="random"):
    dataset=TrainingDataset(image_folder, transform=transform, tokenizer=tokenizer, select=select, max_concept_length=max_concept_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader


def parse_concept(input_concept):
    """
    parse the input concept into a list of concepts for evaluation, supported formats:
    concept: str:
        'man' -> ['man'] (generate an image of a man)
    concept: str:
        'man,young' -> ['man', 'young'] (generate an image of a young man)
    concept: list[str]:
        ['man', 'woman'] -> ['man'], ['woman'] (generate two images: a man, a woman)
    concept: list[str]:
        ['man,young', 'woman,young'] -> [['man','young'],['woman','young']] (generate two images: a young man, an old woman)

    The output of this function is directly fed to int_to_onehot and return a multi-hot vector which can be directly used by the model
    """
    def parse_concept_string(concept):
        assert isinstance(concept, str)
        concept = concept.split(',')
        concept = [x.strip() for x in concept]
        return concept
    
    if isinstance(input_concept, str):
        input_concept = parse_concept_string(input_concept)
        input_concept = [input_concept]

    elif isinstance(input_concept, list):
        input_concept = [parse_concept_string(x) for x in input_concept]

    else:
        raise ValueError(input_concept)
    
    return input_concept


def get_test_data(data_dir, given_prompt=None, given_concept=None, with_baseline=True, device='cuda', max_concept_length=100):
    """
    data_dir: path to data file
    prompt: str
    concept: str or list[str]
    """
    concept_dict=json.load(open(data_dir+'/concept_dict.json','r'))
    if not given_prompt or not given_concept:
        prompt, concept=json.load(open(data_dir+'/test.json','r'))
    if given_prompt:
        prompt=given_prompt
    if given_concept:
        concept=given_concept

    concept = parse_concept(concept)
    print(f'eval with concept: {concept}')

    concept=[int_to_onehot([concept_dict[c_i] for c_i in c], max_concept_length).to(device).unsqueeze(0) for c in concept]
    if with_baseline:
        concept.insert(0, None)
    prompt = [prompt] * len(concept)
    return prompt, concept


def get_i2p_data(data_dir=None, given_prompt=None, given_concept=None, with_baseline=True, device='cuda', max_concept_length=100):
    import pandas as pd

    i2p = pd.read_csv("./i2p_benchmark.csv")
    if given_prompt:
        prompts=i2p[i2p.categories.apply(lambda x: given_prompt in x)].prompt.values.tolist()
    else:
        prompts = i2p.prompt.values.tolist()

    concept_label=[given_concept] if isinstance(given_concept, str) else given_concept
    concept_dict=json.load(open(data_dir+'/concept_dict.json','r'))
    concept=[int_to_onehot(concept_dict[x], max_concept_length).to(device).unsqueeze(0) for x in concept_label]
    if with_baseline:
        concept.insert(0, None)
        concept_label.insert(0, 'none')

    inputs = []
    for prompt in prompts:
        for c_i, c in zip(concept, concept_label):
            inputs.append([prompt, c_i, c])
    return inputs
