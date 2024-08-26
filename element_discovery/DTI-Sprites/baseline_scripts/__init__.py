from .cosegmentation import WeizmannHorseDataset
from .gtsrb import GTSRB8Dataset
from .multi_object import (
    DSpritesGrayDataset,
    TetrominoesDataset,
    CLEVR6Dataset,
    FleuronsCompSyntDataset,
    FleuronsCompDataset,
)
from .instagram import InstagramDataset
from .torchvision import SVHNDataset


def get_dataset(dataset_name):
    return {
        # Cosegmentation
        "weizmann_horse": WeizmannHorseDataset,
        # Custom
        "gtsrb8": GTSRB8Dataset,
        "instagram": InstagramDataset,
        # MultiObject
        "clevr6": CLEVR6Dataset,
        "dsprites_gray": DSpritesGrayDataset,
        "tetrominoes": TetrominoesDataset,
        "fleuron_compounds_synt": FleuronsCompSyntDataset,
        "fleuron_compounds": FleuronsCompDataset,
        # Torchvision
        "svhn": SVHNDataset,
    }[dataset_name]
