from collections.abc import Callable
from typing import cast
import torch
try:
    import timm
    #import uni as _  # noqa: F401
    from PIL.Image import Image
    from timm.data.config import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torch import Tensor
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "uni dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[uni]'`"
    ) from e

from stamp.preprocessing.extractor import Extractor

# pretrained=True needed to load UNI2-h weights (and download weights for the first time)

def uni2() -> Extractor:
    timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    transform = cast(
        Callable[[Image], Tensor],
        create_transform(**resolve_data_config(model.pretrained_cfg, model=model)),
    )
    model.eval()

    return Extractor(
            model=model,
            transform=transform,
            identifier="uni2",  # type: ignore
        )