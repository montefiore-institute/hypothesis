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

        min_x = np.amin(-4.95)
        max_x = np.amax(4.95)
        x_range = np.linspace(float(min_x), float(max_x), 1000)

        y_min = []
        y_max = []

        #also get the theta, which produces the largest gap
        max_gap = 0
        max_gap_theta = None
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
            if tmp_y_max - tmp_y_min > max_gap:
                max_gap = tmp_y_max - tmp_y_min
                max_gap_theta = x
            y_min.append(tmp_y_min)
            y_max.append(tmp_y_max)

        #then get classifiers, which produce min / max output
        print("{} reference: maximum gap ({}) produced at theta {}.".format(ref, max_gap, max_gap_theta))
        min_output, max_output = None, None
        min_classifier_index, max_classifier_index = None, None

        theta = [[max_gap_theta]] * 10
        theta = torch.Tensor(theta)
        input = torch.cat([theta, observations], dim=1).detach()

        for i in range(len(classifiers)):
            s = classifiers[i](input).log().sum().exp().item()
            if min_output is None or min_output > s:
                min_output = s
                min_classifier_index = i
            if max_output is None or max_output < s:
                max_output = s
                max_classifier_index =i
        print("classifier indexes: {} and {}".format(min_classifier_index, max_classifier_index))
        print("##############################")

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
        plt.savefig("plots/{}_ref.png".format(ref))

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
        N = Normal(-4.5, 1.)
        observations = N.sample(torch.Size([10, 1]))
        torch.save(observations, path)
    else:
        observations = torch.load(path)

    return observations

if __name__ == "__main__":
    main()
