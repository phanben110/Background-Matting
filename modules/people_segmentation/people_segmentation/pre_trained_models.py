from collections import namedtuple
from torch import nn
from torch.utils import model_zoo
from modules.iglovikov_helper_functions.dl.pytorch.utils import rename_layers
import torch

from segmentation_models_pytorch import Unet

model = namedtuple("model", ["url", "model"])

models = {
    "Unet_2020-07-20": model(
        url="https://github.com/ternaus/people_segmentation/releases/download/0.0.1/2020-09-23a.zip",
        model=Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None),
    )
}


def create_model(model_name: str) -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")["state_dict"]
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)
    #model.load_state_dict(torch.load('/home/pcwork/ai/ftech/finger/people_segmentation/2020-09-23a.pth',map_location="cuda:0"),strict=False)

    return model
