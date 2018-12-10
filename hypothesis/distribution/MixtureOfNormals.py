from torch.distributions.uniform import Uniform

class MixtureOfNormals:

    def __init__(self, normals, mixing_coefficients=None):
        self.normals = normals
        if(mixing_coefficients == None):
            self.mixing_coefficients = [1.0/len(normals)]*len(normals)
        else:
            self.mixing_coefficients = mixing_coefficients

        self.component_thresholds = [0]
        for x in self.mixing_coefficients[:-1]:
                self.component_thresholds.append(self.component_thresholds[-1] + x)
        self.component_thresholds.reverse()

    def prob(self, x):
        prob = 0

        for i in range(len(self.normals)):
            prob += self.mixing_coefficients[i] * \
                              self.normals[i].log_prob(x).exp()

        return prob

    def log_prob(self, x):
        return self.prob(x).log()

    def sample(self):
        threshold = Uniform(0, 1).sample()

        component_index = 0

        for i in range(len(self.component_thresholds)):
            if(threshold >= self.component_thresholds[i]):
                component_index = len(self.normals) - 1 - i
                break

        return self.normals[component_index].sample()
