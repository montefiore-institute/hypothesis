import torch



class LeNet(torch.nn.Module):

    def __init__(self, activation=torch.nn.ReLU, batchnorm=True, trunk=(256, 256, 256, 1)):
        super(LeNet, self).__init__()
        self.activation = activation(inplace=True)
        self.head_conv_1 = torch.nn.Conv2d(1, 6, 5)
        self.head_conv_2 = torch.nn.Conv2d(6, 16, 5)
        self.latent_dimensionality = 100
        layers = []
        layer = torch.nn.Linear(self.latent_dimensionality, trunk[0])
        for index in range(1, len(trunk)):
            layers.append(self.activation)
            layer = torch.nn.Linear(trunk[index - 1], trunk[index])
            layers.append(layer)
        self.trunk = torch.nn.Sequential(*layers)

    def _compute_head(self, xs):
        zs = self.activation(self.head_conv_1(xs))
        zs = F.max_pool2d(zs, 2)
        zs = self.activation(self.head_conv_2(zs))
        zs = F.max_pool2d(zs, 2)
        zs = zs.view(zs.shape[0], -1) # Flatten

        return zs

    def _compute_trunk(self, zs):
        return self.trunk(zs)

    def log_ratio(self, xs):
        zs = self._compute_head(xs)
        ys = self._compute_trunk(zs)

        return ys

    def forward(self, xs):
        return self.log_ratio(xs).sigmoid()
