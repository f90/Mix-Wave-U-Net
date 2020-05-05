import numpy as np
from sacred import Ingredient

config_ingredient = Ingredient("cfg")

@config_ingredient.config
def cfg():
    # Base configuration
    model_config = {"enst_path" : "/mnt/windaten/Datasets/ENST_Drums", # SET ENST PATH HERE
                    "estimates_path" : "/mnt/windaten/Source_Estimates", # SET THIS PATH TO WHERE YOU WANT OUTPUTS PRODUCED BY THE TRAINED MODEL TO BE SAVED. Folder itself must exist!
                    "data_path" : "/mnt/windaten/Mix-U-Net Data", # Set this to where the preprocessed dataset should be saved

                    "model_base_dir" : "checkpoints", # Base folder for model checkpoints
                    "log_dir" : "logs", # Base folder for logs files
                    "batch_size" : 16, # Batch size
                    "lr" : 1e-4, # Learning rate
                    "epoch_it" : 2000, # Number of update steps per epoch
                    'cache_size': 1000, # Number of audio snippets buffered in the random shuffle queue. Larger is better, since workers put multiple examples of one song into this queue. The number of different songs that is sampled from with each batch equals cache_size / num_snippets_per_track. Set as high as your RAM allows.
                    'num_workers' : 4, # Number of processes used for each TF map operation used when loading the dataset
                    "num_snippets_per_track" : 100, # Number of snippets that should be extracted from each song at a time after loading it. Higher values make data loading faster, but can reduce the batches song diversity
                    'num_layers' : 10, # How many U-Net layers
                    'filter_size' : 15, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 5, # For Wave-U-Net: Filter size of conv in upsampling block
                    'input_filter_size' : 15, # For Wave-U-Net: Filter size of first convolution in first downsampling block
                    'output_filter_size': 1, # For Wave-U-Net: Filter size of first convolution in first downsampling block
                    'num_initial_filters' : 24, # Number of filters for convolution in first layer of network
                    "num_frames": 16384, # DESIRED number of time frames in the output waveform per samples (could be changed when using valid padding)
                    'expected_sr': 44100,  # Downsample all audio input to this sampling rate
                    'mono_downmix': False,  # Whether to downsample the audio input
                    'output_type' : 'direct', # Type of output layer. Direct output: Linear layer without activation
                    'output_activation' : 'linear', # Activation function for output layer. "tanh" or "linear". Linear output involves clipping to [-1,1] at test time, and might be more stable than tanh
                    'context' : False, # Type of padding for convolutions in model. If False, feature maps double or half in dimensions after each convolution, and convolutions are padded with zeros ("same" padding). If True, convolution is only performed on the available mixture input, thus the output is smaller than the input
                    'network' : 'unet', # Type of network architecture
                    'upsampling' : 'linear', # Type of technique used for upsampling the feature maps in a unet architecture, either 'linear' interpolation or 'learned' filling in of extra samples
                    'task' : 'dry', # Type of separation task. 'voice' : Separate music into voice and accompaniment. 'multi_instrument': Separate music into guitar, bass, vocals, drums and other (Sisec)
                    'augmentation' : False, # Random attenuation of input signals to improve generalisation performance (data augmentation)
                    'worse_epochs' : 20, # Patience for early stoppping on validation set
                    }
    experiment_id = np.random.randint(0,1000000)
        
    model_config["input_names"] = ['hi-hat', 'kick', 'overhead_L', 'overhead_R', 'snare', 'tom_1', 'tom_2', 'tom_3']
    model_config["num_inputs"] = len(model_config["input_names"])
    model_config["num_outputs"] = 1 if model_config["mono_downmix"] else 2

@config_ingredient.named_config
def context_wet():
    print("Training wet model")
    model_config = {
        "task": "wet",
        "num_frames": 88200,
        "context": True
    }

@config_ingredient.named_config
def context_dry():
    print("Training dry model")
    model_config = {
        "num_frames": 88200,
        "context": True
    }
