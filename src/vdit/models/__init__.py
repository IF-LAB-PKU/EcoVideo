from vdit.models.EDEN import EDEN
from vdit.models.EDEN_VAE import VAE
from vdit.models.EDEN_DiT import DiT


def load_model(model_name, **model_args):
    if model_name == "EDEN":
        return EDEN(**model_args)
    elif model_name == "EDEN_VAE":
        return VAE(**model_args)
    elif model_name == "EDEN_DiT":
        return DiT(**model_args)
    else:
        raise ValueError(f"No model named {model_name} in models!")
