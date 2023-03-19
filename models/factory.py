import os

from .registry import model_entrypoint, is_model


def create_model(
        model_name: str,
        pretrained: bool = False,
        checkpoint_path='',
        pretrained_version=None,
        **kwargs
        ):
    """Create a model
    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized
    Keyword Args:
        **: other kwargs are model specific
    """

    kwargs = {k: v for k, v in kwargs.items() if v is not None and k != 'name'}

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoint(model_name)
    if pretrained_version:
        model = create_fn(pretrained=pretrained, pretrained_version=pretrained_version, **kwargs)
    else:
        model = create_fn(pretrained=pretrained, **kwargs)

    if os.path.isfile(checkpoint_path):
        model.load_checkpoint(model, checkpoint_path)
    return model
