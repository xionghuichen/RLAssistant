# Created by xionghuichen at 2022/6/22
# Email: chenxh@lamda.nju.edu.cn

import torch as th
from torch import nn
from typing import Dict, List, Tuple, Type, Union



def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return:
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")
    return device

def to_tensor(x, device="auto"):
    return th.as_tensor(x).to(get_device('auto'))

class MLP(nn.Module):
    def __init__(self, feature_dim: int, net_arch: List[int], activation_fn: Type[nn.Module],
                 device: Union[th.device, str] = "auto"):
        super(MLP, self).__init__()
        device = get_device(device)
        net = []
        last_layer_dim_shared = feature_dim
        for layer in net_arch:
            net.append(nn.Linear(last_layer_dim_shared, layer))  # add linear of size layer
            net.append(activation_fn())
            last_layer_dim_shared = layer
        net.append(nn.Linear(last_layer_dim_shared, 1))
        self.net = nn.Sequential(*net).to(device)


    def forward(self, features: th.Tensor) -> th.Tensor:
        return self.net(features)


