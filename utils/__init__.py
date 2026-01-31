from torch import nn

class WineQualityFNN(nn.Module):

    def __init__(self,
                 input_dim, ):
        super().__init__()