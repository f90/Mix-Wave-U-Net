# Mix-Wave-U-Net
Implementation of the [Mix-Wave-U-Net](https://arxiv.org/abs/1806.03185) for automatic mixing.

## Listening examples

Listen to vocal separation results [here](audio_examples/outputs/).

## What is the Mix-Wave-U-Net?
The Wave-U-Net is a convolutional neural network applicable to audio source separation tasks, which works directly on the raw audio waveform, presented in [this paper](https://arxiv.org/abs/1806.03185).

<TODO Mix-Wave-U-Net is....> 

See the diagram below for a summary of the network architecture.

<img src="./mixwaveunet.png" width="500">

# Installation

## Requirements

GPU strongly recommended to avoid very long training times (CUDA 10.0 required)

The project is based on Python 3.6 and requires [libsndfile](http://mega-nerd.com/libsndfile/) to be installed.

Then, the following Python packages need to be installed:

```
numpy==1.18.3
sacred==0.8.1
tensorflow-gpu==1.15.2
librosa==0.7.2
soundfile==0.10.3.post1
lxml==4.5.0
google
protobuf
soundfile
tqdm
```

Instead of ``tensorflow-gpu``, the CPU version of TF, ``tensorflow`` can be used, if there is no GPU available.

All the above packages are also saved in the file ``requirements.txt`` located in this repository, so you can clone the repository and then execute the following in the downloaded repository's path to install all the required packages at once:

``pip install -r requirements.txt``

### Download datasets

To directly use the pre-trained models we provide for download to separate your own songs, now skip directly to the [last section](#test), since the dataset is not needed in that case.

To reproduce the experiments in the paper (train all the models), you need to download the [ENST dataset](https://sigsep.github.io/datasets/musdb.html) and extract it into a folder of your choice. It should then have "drummer_1", "drummer_2" and "drummer_3" subfolders in it.

### Set-up filepaths

Now you need to set up the correct file paths for the dataset and the location where outputs should be saved.

Open the ``Config.py`` file, and set the ``enst_path`` entry of the ``model_config`` dictionary to the location of the main folder of the ENST dataset.
Also set the ``estimates_path`` entry of the same ``model_config`` dictionary to the path pointing to an empty folder where you want the final model outputs to be saved into.

## Training the models / model overview

Since the paper investigates many model variants of the Wave-U-Net and also trains the [U-Net proposed for vocal separation](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf), which achieved state-of-the-art performance, as a comparison, we give a list of model variants to train and the command needed to start training them:

| Model name (from paper) | Description                                             | Separate vocals or multi-instrument? | Command for training                          |
|-------------------------|---------------------------------------------------------|--------------------------------------|-----------------------------------------------|
| M1                      | Baseline Wave-U-Net model                               | Vocals                               | ``python Training.py``                            |
| M2                      | M1 + difference output layer                            | Vocals                               | ``python Training.py with cfg.baseline_diff``         |

# <a name="test"></a> Test trained models on songs!

We provide a pretrained versions of models M4, M6 and M5-HighSR so you can separate any of your songs right away. 

## Downloading our pretrained models

Download our pretrained models [here](https://www.dropbox.com/s/oq0woy3cmf5s8y7/models.zip?dl=1).
Unzip the archive into the ``checkpoints`` subfolder in this repository, so that you have one subfolder for each model (e.g. ``REPO/checkpoints/baseline_stereo``)

## Run pretrained models

For a quick demo on an example song with our pre-trained best vocal separation model (M5-HighSR), one can simply execute

`` python Predict.py with cfg.full_44KHz ``

to separate the song "Mallory" included in this repository's ``audio_examples`` subfolder into vocals and accompaniment. The output will be saved next to the input file.

To apply our pretrained model to any of your own songs, simply point to its audio file path using the ``input_path`` parameter:

`` python Predict.py with cfg.full_44KHz input_path="/mnt/medien/Daniel/Music/Dark Passion Play/Nightwish - Bye Bye Beautiful.mp3"``