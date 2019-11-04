import torch



class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, shape_xs, shape_ys,
        layers=(128, 128), activation=torch.nn.ELU,
        transform_output="normalize"):
        super(MultiLayerPerceptron, self).__init__()
        self.dimensionality_xs = 1
        self.dimensionality_ys = 1
        self.shape_xs = shape_xs
        self.shape_ys = shape_ys
        self._compute_dimensionality()
        mappings = []
        mappings.append(torch.nn.Linear(self.dimensionality_xs, layers[0]))
        for layer_index in range(len(layers) - 1):
            mappings.append(activation())
            current_layer = layers[layer_index]
            next_layer = layers[layer_index + 1]
            mappings.append(torch.nn.Linear(
                current_layer, next_layer))
        mappings.append(activation())
        mappings.append(torch.nn.Linear(layers[-1], self.dimensionality_ys))
        if transform_output is "normalize":
            if self.dimensionality_ys > 1:
                layer = torch.nn.Softmax(dim=0)
            else:
                layer = torch.nn.Sigmoid()
            mappings.append(layer)
        elif transform_output is not None:
            mappings.append(transform_output())
        self.network = torch.nn.Sequential(*mappings)

    def _compute_dimensionality(self):
        for shape_element in self.shape_xs:
            self.dimensionality_xs *= shape_element
        for shape_element in self.shape_ys:
            self.dimensionality_ys *= shape_element

    def forward(self, xs):
        xs = xs.view(-1, self.dimensionality_xs)

        return self.network(xs)
