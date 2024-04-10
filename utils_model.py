import torch

"""Based on the logic that only layers that requires grad are modified and thus saved"""


def load_model(model, path):
    checkpoint=torch.load(path)
    _count_parameters(checkpoint)
    model=_load_partial_state_dict(model, checkpoint)
    return model

def save_model(model, path):
    _count_trainable_parameters(model)
    keys = [k for k,v in model.named_parameters() if v.requires_grad]
    keys = set(keys)
    model_dict = model.state_dict()
    model_dict = {k:v for k,v in model_dict.items() if k in keys}
    torch.save(model_dict, path)

def load_weights(model, path):
    checkpoint=torch.load(path)
    model=_load_partial_state_dict(model, checkpoint)
    return model

def _count_trainable_parameters(model):
    total=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num trainable params: ", total)
    return total

def _count_parameters(state_dict):
    total=sum(p.numel() for p in state_dict.values())
    print("Num trainable params: ", total)
    return total

def _load_partial_state_dict(model, checkpoint):
    """checkpoint has to totally match a subset of the model state dict"""
    model_dict = model.state_dict()
    model_dict.update(checkpoint)
    model.load_state_dict(model_dict)
    return model
