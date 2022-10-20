import torch
from RLA.easy_log.tester import tester


def save_model_dict(model_dict):
    torch.save(model_dict, f=tester.checkpoint_dir + "checkpoint.pt")

def load_checkpoint(tester):
    ckpt_dict = torch.load(tester.checkpoint_dir + "checkpoint.pt")
    return ckpt_dict