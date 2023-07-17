import segmentation_models_pytorch as smp


def get_mt_bisenet(**kwargs):
    return smp.Mt_bisenet(**kwargs)

_models = {
    'mt_bisenet':get_mt_bisenet,
}

def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net