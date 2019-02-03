# Predefined hooks.
post_inference    = "post_inference"
post_simulation   = "post_simulation"
post_step         = "post_step"
pre_inference     = "pre_inference"
pre_simulation    = "pre_simulation"
pre_step          = "pre_step"
exception         = "exception"
reset             = "reset"
pre_epoch         = "pre_epoch"
post_epoch        = "post_epoch"
pre_validation    = "pre_validation"
post_validation   = "post_validation"
pre_checkpoint    = "pre_checkpoint"
post_checkpoint   = "post_checkpoint"
pre_reset         = "pre_reset"
post_reset        = "post_reset"
end               = "end"

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
