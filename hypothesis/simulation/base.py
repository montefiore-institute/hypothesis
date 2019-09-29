import torch



class Simulator(torch.nn.Module):
    """
    """

    def __init__(self):
        super(Simulator, self).__init__()

    def forward(self, inputs):
        raise NotImplementedError

    def terminate(self):
        pass
