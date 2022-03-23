from matplotlib import use
import torch
import torch.nn as nn

class TestTimeAdaptationModule(nn.Module):
    def __init__(self, task_module, i2n_module, use_tta):
        super(TestTimeAdaptationModule, self).__init__()
        self.task_module = task_module
        self.i2n_module = i2n_module
        self.use_tta = use_tta

    def forward(self, use_tta=False):
        if not use_tta:
            normalised_output = self.i2n_module()
