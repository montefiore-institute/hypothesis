# Predefined hooks.
pre_step          = "pre_step"
post_step         = "post_step"
reset             = "reset"
pre_simulation    = "pre_simulation"
post_simulation   = "post_simulation"
pre_inference     = "pre_inference"
post_inference    = "post_inference"

# Hook storage.
hooks = {}


def call_hooks(tag, argument, **kwargs):
    if tag in hooks.keys() and len(hooks[tag]) > 0:
        for f in hooks[tag]:
            f(argument, **kwargs)


def register_hook(tag, f):
    if tag not in hooks.keys():
        hooks[tag] = []
    if f not in hooks[tag]:
        hooks[tag].append(f)


def clear_hooks(tag=None):
    if tag:
        del hooks[tag]
    else:
        keys = list(hooks.keys())
        for key in keys:
            del hooks[key]
