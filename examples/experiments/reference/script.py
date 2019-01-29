import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from hypothesis.util import epsilon
from torch.distributions.normal import Normal



def main():
    for ref in ["w", "wo"]:
        classifiers = load_classifiers(ref)
        observations = get_observations()

        min_x = np.amin(-5)
        max_x = np.amax(5)
        x_range = np.linspace(float(min_x), float(max_x), 1000)

        y_min = []
        y_max = []

        for x in x_range:
            tmp_y_min, tmp_y_max = None, None
            theta = [[x]] * 10
            theta = torch.Tensor(theta)
            input = torch.cat([theta, observations], dim=1).detach()
            for classifier in classifiers:
                s = classifier(input).log().sum().exp().item()
                if tmp_y_min is None or tmp_y_min > s:
                    tmp_y_min = s
                if tmp_y_max is None or tmp_y_max < s:
                    tmp_y_max = s
            y_min.append(tmp_y_min)
            y_max.append(tmp_y_max)

        # variance version
        #for x in x_range:
        #    ys = []
        #    theta = [[x]] * 10
        #    theta = torch.Tensor(theta)
        #    input = torch.cat([theta, observations], dim=1).detach()
        #    for classifier in classifiers:
        #        s = classifier(input).log().sum().exp().item()
        #        ys.append(s)
        #    y_mean = np.mean(ys)
        #    y_var = np.var(ys)
        #    y_min.append(y_mean-y_var)
        #    y_max.append(y_mean+y_var)

        plt.figure()
        plt.fill_between(x_range, y_min, y_max)
        plt.savefig("{}_ref.png".format(ref))

def load_classifiers(ref):
    classifiers = []
    for i in range(0, 16):
        classifier = torch.load("models_{}_reference/run{}_hidden500_final.th".format(ref, i))
        classifier.eval()
        classifiers.append(classifier)
    return classifiers

def get_observations():
    path = "observations/"
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + "observations.th"
    if not os.path.exists(path):
        N = Normal(-5., 1.)
        observations = N.sample(torch.Size([10, 1]))
        torch.save(observations, path)
    else:
        observations = torch.load(path)

    return observations

if __name__ == "__main__":
    main()
