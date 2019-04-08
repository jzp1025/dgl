from .parallel_apply import parallel_apply
from .replicate import replicate
from .dgl_data_parallel import DataParallel
from .scatter_gather import scatter, gather
#from .distributed import DistributedDataParallel
#from .distributed_cpu import DistributedDataParallelCPU
import torch.nn.parallel.deprecated

__all__ = ['replicate', 'scatter', 'parallel_apply', 'gather', 'dgl_data_parallel',
           'DataParallel', 'DistributedDataParallel', 'DistributedDataParallelCPU']
