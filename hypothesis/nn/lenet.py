import torch



class LeNet(torch.nn.Module):

    def __init__(self, shape_xs, shape_ys,
                 activation=torch.nn.ReLU,
                 batchnorm=True,
                 trunk=(256, 256, 256),
                 transform_output="normalize"):
        super(LeNet, self).__init__()
        self.activation = activation(inplace=True)
        self.head_conv_1 = torch.nn.Conv2d(1, 6, 5)
        self.head_conv_2 = torch.nn.Conv2d(6, 16, 5)
        self.shape_xs = shape_xs
        self.shape_ys = shape_ys
        self.dimensionality_ys = 1
        self._compute_dimensionality()
        self.latent_dimensionality = None
        mappings = []
        layer = torch.nn.Linear(self.latent_dimensionality, trunk[0])
        mappings.append(layer)
        for index in range(1, len(trunk)):
            mappings.append(self.activation)
            layer = torch.nn.Linear(trunk[index - 1], trunk[index])
            mappings.append(layer)
        mappings.append(self.activation)
        mappings.append(torch.nn.Linear(trunk[-1], self.dimensionality_ys))
        if transform_output is "normalize":
            if self.dimensionality_ys > 1:
                layer = torch.nn.Softmax(dim=0)
            else:
                layer = torch.nn.Sigmoid()
            mappings.append(layer)
        elif transform_output is not None:
            mappings.append(transform_output())
        self.trunk = torch.nn.Sequential(*layers)

    def _compute_dimensionality(self):
        for shape_element in self.shape_ys:
            self.dimensionality_ys *= shape_element

    def _forward_head(self, xs):
        zs = self.activation(self.head_conv_1(xs))
        zs = F.max_pool2d(zs, 2)
        zs = self.activation(self.head_conv_2(zs))
        zs = F.max_pool2d(zs, 2)
        zs = zs.view(zs.shape[0], -1) # Flatten

        return zs

    def _forward_trunk(self, zs):
        return self.trunk(zs)

    def log_ratio(self, xs):
        zs = self._forward_head(xs)
        ys = self._forward_trunk(zs)
        zs = zs.view(self.shape_ys)

        return ys

    def forward(self, xs):
        return self.log_ratio(xs).sigmoid()
