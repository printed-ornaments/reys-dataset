# configuration file for background model
# All path data should be updated according to the user's filesystem
# dataset_name to be updated with the name of the active dataset


class Background_training_configuration_data:

    def __init__(self, dataset_name, kwargs):

        self.dataset_name = "VIGNETTES"
        self.image_height = None
        self.image_width = None
        self.full_dataset_path = None
        self.train_dataset_path = None
        self.test_dataset_path = None
        self.GT_train_dataset_path = None  # ground truth masks path
        self.GT_test_dataset_path = None
        self.results_dir_path = "../results/VIGNETTES"
        self.use_trained_model = False
        self.saved_model_path = None
        self.unsupervised_mode = False  # the number of iterations and background complexity have to be set manually
        self.n_iterations = 500000
        self.background_complexity = True
        self.train_model = True
        self.learning_rate = 2e-3
        self.weight_decay = 0
        self.batch_size = 64
        self.device = "cuda"

        # path where temporary training samples are stored during training
        self.training_images_output_directory = "../AST/FULLVIG/"

        # standard AE-NE hyperparameters
        self.beta = 6
        self.r = 75
        self.tau_0 = 0.24
        self.tau_1 = 0.25
        self.alpha_1 = 96 / 255
        self.alpha_2 = 7.0

        # override default values with arguments provided at init:
        for attr, value in kwargs.items():
            assert hasattr(self, attr), f"wrong attribute name {attr}"
            setattr(self, attr, value)


background_config = {
    "VIGNETTES": {
        "image_height": 128,
        "image_width": 128,
        "n_iterations": 1,
        "background_complexity": True,
        "train_dataset_path": "../../crop_128x128",  # "/path/to/dataset",
        "results_dir_path": "../outputs",
        "saved_model_path": "./outputs/models/trained_model.pth",
    }
}
dataset_name = "VIGNETTES"

env = Background_training_configuration_data(
    dataset_name, background_config[dataset_name]
)
