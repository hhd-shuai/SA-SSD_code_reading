from .conv_module import ConvModule
from .norm import build_norm_layer
from .weight_init import (xavier_init, normal_init, uniform_init, kaiming_init,
                          bias_init_with_prob)
from .empty import Empty
from .sequential import Sequential
import inspect
import torch

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]
    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

def get_pos_to_kw_map(func):
    pos_to_kw = {}
    # inspect.signature函数返回一个inspect.Signature对象，它有一个parameters属性，这是一个有序映射，把参数名和inspect.Parameter对象对应起来，各个Paramters属性他有自己的属性
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        # POSITIONAL_OR_KEYWORD 可以通过定位参数和关键字参数传入的形参
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw # {0: 'self', 1: 'in_channels', 2: 'out_channels', 3: 'kernel_size', 4: 'stride', 5: 'padding', 6: 'dilation', 7: 'groups', 8: 'bias', 9: 'padding_mode'}

# eg: BatchNorm2d = change_default_args(
#             eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
def change_default_args(**kwargs):  # kwargs={dict:2} {'eps': 0.001, 'momentum': 0.01}
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class): # layer_class <class 'torch.nn.modules.batchnorm.BatchNorm2d'>
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__) # {0: 'self', 1: 'in_channels', 2: 'out_channels', 3: 'kernel_size', 4: 'stride', 5: 'padding', 6: 'dilation', 7: 'groups', 8: 'bias', 9: 'padding_mode'}
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()} # {'self': 0, 'in_channels': 1, 'out_channels': 2, 'kernel_size': 3, 'stride': 4, 'padding': 5, 'dilation': 6, 'groups': 7, 'bias': 8, 'padding_mode': 9}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper

def one_hot(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(
        *list(tensor.shape), depth, dtype=dtype, device=tensor.device)
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot

__all__ = [
    'ConvModule', 'build_norm_layer', 'xavier_init', 'normal_init',
    'uniform_init', 'kaiming_init', 'bias_init_with_prob','Empty',
    'change_default_args','Sequential','one_hot', 'get_paddings_indicator'
]
