import torch



class Ensemble(torch.nn.Module):

    def __init__(self, classifiers):
        super(Ensemble, self).__init__()
        self.classifiers = classifiers

    def forward(self, inputs):
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(inputs))
        output = torch.cat(outputs, dim=0).mean(dim=0)

        return output
