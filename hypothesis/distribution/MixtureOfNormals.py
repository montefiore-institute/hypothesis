class MixtureOfNormals:

    def __init__(self, normals, mixing_coefficients=None):
        self.normals = normals
        if(mixing_coefficients == None):
            self.mixing_coefficients = [1.0/len(normals)]*len(normals)
        else:
            self.mixing_coefficients = mixing_coefficients

    def prob(self, x):
        prob = 0

        for i in range(len(self.normals)):
            prob += self.mixing_coefficients[i] * \
                              self.normals[i].log_prob(x).exp()

        return prob

    def log_prob(self, x):
        return self.prob(x).log()
