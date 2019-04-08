import operator
import torch
import warnings
from itertools import chain
from torch.nn.modules import Module
from .scatter_gather import scatter_g_features, gather
from .replicate import replicate
from .parallel_apply import parallel_apply
from torch.cuda._utils import _get_device_index

def _check_balance(device_ids):
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""


class DataParallel(Module):
    
    def __init__(self, module, g, features, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

#        print("init")
        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device("cuda:{}".format(self.device_ids[0]))

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, g, features):
        print("forward :")
        print("device_ids: ", self.device_ids)
        print("g: ", g)
        print("features: ", features.size())
        print("-------------------")
        if not self.device_ids:
            return self.module(g, features)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        #inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(g, features)
        #replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        #outputs = self.parallel_apply(replicas, inputs, kwargs)
        #return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, g, features, device_ids):
        return scatter_g_features(g, features, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, g, features):
        return parallel_apply(replicas, g, features, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


"""

def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    Evaluates module(input) in parallel across the GPUs given in device_ids.
    This is the functional version of the DataParallel module.
    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    output_device = _get_device_index(output_device, True)
    src_device_obj = torch.device("cuda:{}".format(device_ids[0]))

    for t in chain(module.parameters(), module.buffers()):
        if t.device != src_device_obj:
            raise RuntimeError("module must have its parameters and buffers "
                               "on device {} (device_ids[0]) but found one of "
                               "them on device: {}".format(src_device_obj, t.device))

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)
"""
