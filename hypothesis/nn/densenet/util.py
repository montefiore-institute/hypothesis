import torch



def load_modules(dimensionality):
    configurations = {
        1: load_modules_1_dimensional,
        2: load_modules_2_dimensional,
        3: load_modules_3_dimensional}

    return configurations[dimensionality]()


def load_modules_1_dimensional():
    c = torch.nn.Conv1d
    b = torch.nn.BatchNorm1d
    m = torch.nn.MaxPool1d
    a = torch.nn.AvgPool1d
    ap = torch.nn.AdaptiveAvgPool1d

    return c, b, m, a, ap


def load_modules_2_dimensional():
    c = torch.nn.Conv2d
    b = torch.nn.BatchNorm2d
    m = torch.nn.MaxPool2d
    a = torch.nn.AvgPool2d
    ap = torch.nn.AdaptiveAvgPool2d

    return c, b, m, a, ap


def load_modules_3_dimensional():
    c = torch.nn.Conv3d
    b = torch.nn.BatchNorm3d
    m = torch.nn.MaxPool3d
    a = torch.nn.AvgPool3d
    ap = torch.nn.AdaptiveAvgPool3d

    return c, b, m, a, ap


def load_configuration_121():
    growth_rate = 32
    input_features = 64
    config = [6, 12, 24, 16]

    return growth_rate, input_features, config


def load_configuration_161():
    growth_rate = 48
    input_features = 96
    config = [6, 12, 36, 24]

    return growth_rate, input_features, config


def load_configuration_169():
    growth_rate = 32
    input_features = 64
    config = [6, 12, 32, 32]

    return growth_rate, input_features, config


def load_configuration_201():
    growth_rate = 32
    input_features = 64
    config = [6, 12, 48, 32]

    return growth_rate, input_features, config
