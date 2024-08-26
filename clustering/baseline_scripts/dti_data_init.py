from .raw import LettersDataset, GenericDataset


def get_dataset(dataset_name):
    if dataset_name == "generic":
        return GenericDataset

    from .cosegmentation import WeizmannHorseDataset
    from .gtsrb import GTSRB8Dataset
    from .affnist import AffNISTTestDataset
    from .fleurons import FleuronsDataset
    from .hdf5 import FRGCDataset
    from .coa import CoADataset
    from .multi_object import (
        DSpritesGrayDataset,
        TetrominoesDataset,
        CLEVR6Dataset,
        FleuronsCompDataset,
        FleuronsCompSyntDataset,
    )
    from .instagram import InstagramDataset
    from .torchvision import (
        SVHNDataset,
        FashionMNISTDataset,
        MNISTDataset,
        MNISTTestDataset,
        MNISTColorDataset,
        MNIST1kDataset,
        USPSDataset,
    )

    return {
        # Cosegmentation
        "weizmann_horse": WeizmannHorseDataset,
        # Custom
        "affnist_test": AffNISTTestDataset,
        "gtsrb8": GTSRB8Dataset,
        "instagram": InstagramDataset,
        "frgc": FRGCDataset,
        # MultiObject
        "clevr6": CLEVR6Dataset,
        "dsprites_gray": DSpritesGrayDataset,
        "tetrominoes": TetrominoesDataset,
        "fleuron_compounds": FleuronsCompDataset,
        "fleuron_compounds_synt": FleuronsCompSyntDataset,
        # Torchvision
        "fashion_mnist": FashionMNISTDataset,
        "mnist": MNISTDataset,
        "mnist_test": MNISTTestDataset,
        "mnist_color": MNISTColorDataset,
        "mnist_1k": MNIST1kDataset,
        "svhn": SVHNDataset,
        "usps": USPSDataset,
        # Fleurons
        "fleurons": FleuronsDataset,
        # Letters
        "letters": LettersDataset,
        "Lettre_a": LettersDataset,
        "Lettre_e": LettersDataset,
        "Lettre_i": LettersDataset,
        # CoA
        "coa": CoADataset,
        # Generic
        "generic": GenericDataset,
    }[dataset_name]


def get_subset(dataset_name):
    from .fleurons import FleuronsDataset

    if dataset_name == "fleurons":
        return FleuronsDataset
    else:
        raise NotImplementedError()
