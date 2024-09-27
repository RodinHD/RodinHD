import os
from torch.utils.tensorboard import SummaryWriter

class Visualizer(object):
    def __init__(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.writer = SummaryWriter(path)
        
    def Scalar(self, name, val, step):
        self.writer.add_scalar(name, val, step)

    def write_dict(self, dict, step):
        for key, value in dict.items():
            self.writer.add_scalar(key, value, step)
