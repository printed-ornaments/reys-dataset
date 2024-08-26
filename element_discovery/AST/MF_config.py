import torch
import torch.utils.data


class Foreground_training_configuration_data:
    def __init__(self, dataset_name, kwargs):

        self.dataset_name = dataset_name  # dataset name

        ## dataset and checkpoint Paths
        self.test_dataset_input_path = (
            None  # Path to the frames sequence directory used for testing
        )
        self.test_dataset_background_path = None
        self.test_dataset_target_path = (
            None  # Path to the frames sequence directory used for testing
        )
        self.test_dataset_GT_mask_path = None  # Path to the ground truth masks directory used for evaluation during testing
        self.train_dataset_input_path = (
            None  # Path to the frames sequence directory used for testing
        )
        self.train_dataset_background_path = None
        self.train_dataset_target_path = (
            None  # Path to the frames sequence directory used for testing
        )
        self.train_dataset_GT_mask_path = None  # Path to the ground truth masks directory used for evaluation during testing
        self.object_model_checkpoint_path = "MF_outputs/2024-08-10T16:27:14.809950kground_images_train/checkpoint_final_50000_epochs.pth"  # None
        self.results_dir_path = None

        ## videos descriptors
        self.max_set_size = None
        self.image_height = None
        self.image_width = None

        # model training scenario
        self.background_model_training_scenario = "curriculum_training"  # possible scenarii : "random_initialization", "frozen_weights", "curriculum_training"

        # model parameters
        self.feature_map_generator_name = "Segformer"  # "Unet" or "Segformer"
        self.pretrained_backbone_weights_reference = "nvidia/mit-b3"  # for Segformer cf Hugging Face website for available weights
        self.isotropic_scaling = None
        self.transformer_dim = (
            256  # dimension of inputs and outputs of the transformer encoder
        )
        self.transformer_nhead = 8  # number of heads of  transformer encoder
        self.transformer_nlayers = 6  # number of layers of the transformer encoder
        self.transformer_dim_feedforward = 512  # number of layers of  transformer
        self.initial_background_activation_logit = 11
        self.variable_background_activation = False
        self.z_what_dim = None
        self.max_scaling_factor = None
        self.min_scaling_factor = 1.3

        # loss function parameters
        self.pixel_entropy_loss_weight = 5e-5

        # training parameters
        self.batch_size = None
        self.learning_rate = None
        self.number_of_training_steps = None
        self.warmup = None
        self.background_weights_freeze_duration = None
        self.pixel_entropy_loss_full_activation_step = None
        self.workers = 4  # number of workers for data loading per process
        self.object_detection_weight_decay = 0
        self.use_trained_model = (
            False  # if set to true, training will start from the saved checkpoint
        )
        self.device = torch.device(0)

        # user interface parameters
        self.training_images_output_directory = "../MF_outputs/"  # to be updated
        self.n_images_to_show = 6  # number of images to show during training
        self.message_time = 15  # time between two training status message
        self.show_time = 10  # time between images saving (sec)
        self.save_time = 300  # time between model saving
        self.color_palette = None  # default or seaborn

        # override default values with provided arguments:
        for attr, value in kwargs.items():
            assert hasattr(self, attr), f"wrong attribute name {attr}"
            setattr(self, attr, value)


synthetic_images_config = {
    "z_what_dim": 32,
    "batch_size": 15,
    "learning_rate": 1e-5,
    "number_of_training_steps": 200001,
    "warmup": 5000,
    "pixel_entropy_loss_full_activation_step": 1000,
    "background_weights_freeze_duration": 30000,
    "max_scaling_factor": 24,
    "variable_background_activation": False,
    "color_palette": "default",
}

dataset_configs = {
    "VIGNETTES": {
        **synthetic_images_config,
        "image_height": 128,
        "image_width": 128,
        "max_set_size": 10,
        "isotropic_scaling": True,
        "train_dataset_input_path": "./outputs/background_output/background_images_train",
        "train_dataset_background_path": "./outputs/background_output/backgrounds_rgba_train",
        "train_dataset_GT_mask_path": "./outputs/background_output/GT_masks_train",
        "test_dataset_input_path": "./outputs/background_output/background_images_test",
        "test_dataset_background_path": "./outputs/background_output/backgrounds_rgba_test",
        "test_dataset_GT_mask_path": "./outputs/background_output/GT_masks_test",
        "results_dir_path": "./results",
        "use_trained_model": False,
    },
}


dataset_name = "VIGNETTES"


assert dataset_name in ["VIGNETTES"]

args = Foreground_training_configuration_data(
    dataset_name, dataset_configs[dataset_name]
)
