import glob
import os.path
import random
from multiprocessing import Process

import Utils

import numpy as np
import librosa
import soundfile
import os
import tensorflow as tf
import pickle


def take_random_snippets(sample, keys, input_shape, output_shape, num_samples):
    # Take a sample (collection of audio files) and extract snippets from it at a number of random positions
    start_pos = tf.random.uniform([num_samples], 0, maxval=sample["length"] - input_shape[0], dtype=tf.int64)
    return take_snippets_at_pos(sample, keys, start_pos, input_shape[0], output_shape[1], num_samples)

def take_all_snippets(sample, keys, input_shape, output_shape):
    # Take a sample and extract snippets from the audio signals, using a hop size equal to the output size of the network
    start_pos = tf.range(0, sample["length"] - input_shape[0], delta=output_shape[0], dtype=tf.int64)
    num_samples = start_pos.shape[0]
    return take_snippets_at_pos(sample, keys, start_pos, input_shape[0], output_shape[1], num_samples)

def take_snippets_at_pos(sample, keys, start_pos, length, output_channels, num_samples):
    # Take a sample and extract snippets from the audio signals at the given start positions with the given number of samples width
    batch = dict()
    for key in keys:
        
        if key is 'mix':
            
            batch[key] = tf.map_fn(lambda pos: sample[key][pos:pos + length, :], start_pos, dtype=tf.float32)
            batch[key].set_shape([num_samples, length, output_channels])
        else:
            batch[key] = tf.map_fn(lambda pos: sample[key][pos:pos + length, :], start_pos, dtype=tf.float32)
            batch[key].set_shape([num_samples, length, 1])
            
            

    return tf.data.Dataset.from_tensor_slices(batch)

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_records(sample_list, model_config, input_shape, output_shape, records_path):
    # Writes samples in the given list as TFrecords into a given path, using the current model config and in/output shapes

    # Compute padding
    if (input_shape[1] - output_shape[1]) % 2 != 0:
        print("WARNING: Required number of padding of " + str(input_shape[1] - output_shape[1]) + " is uneven!")
    pad_frames = (input_shape[1] - output_shape[1]) // 2

    # Set up writers
    num_writers = 1
    writers = [tf.io.TFRecordWriter(records_path + str(i) + ".tfrecords") for i in range(num_writers)]

    # Go through songs and write them to TFRecords
    all_keys = ["mix"] + model_config["input_names"]
    for sample in sample_list:
        print("Reading song")
        try:
            audio_tracks = dict()

            for key in all_keys:
                
                try:
                    audio, _ = Utils.load(sample[key], sr=model_config["expected_sr"], mono=model_config["mono_downmix"]) 
                    if key is 'mix':
                        lengthMix = audio.shape[0]
                    
                except:
                    audio = np.zeros((lengthMix,1), dtype=np.float32)
                    
                if key is not 'mix':
                    assert(audio.shape[1] == 1)
                else:
                    if not model_config["mono_downmix"]:
                        assert(audio.shape[1] == 2)
                    else:
                        assert(audio.shape[1] == 1)
                        
                    audio = audio/np.max(np.abs(audio))

#                 if not model_config["mono_downmix"] and audio.shape[1] == 1:
#                     print("WARNING: Had to duplicate mono track to generate stereo")
#                     audio = np.tile(audio, [1, 2])


                audio_tracks[key] = audio
        except Exception as e:
            print(e)
            print("ERROR occurred during loading file " + str(sample) + ". Skipping")
            continue

        # Pad at beginning and end with zeros
        audio_tracks = {key : np.pad(audio_tracks[key], [(pad_frames, pad_frames), (0, 0)], mode="constant", constant_values=0.0) for key in audio_tracks.keys()}

        # All audio tracks must be exactly same length and channels
        length = audio_tracks["mix"].shape[0]
        channels = audio_tracks["mix"].shape[1]
        for audio in audio_tracks.values():
            assert(audio.shape[0] == length)

        # Write to TFrecords the flattened version
        feature = {key: _floats_feature(audio_tracks[key]) for key in all_keys}
        feature["length"] = _int64_feature(length)
        feature["channels"] = _int64_feature(channels)
        sample = tf.train.Example(features=tf.train.Features(feature=feature))
        writers[np.random.randint(0, num_writers)].write(sample.SerializeToString())

    for writer in writers:
        writer.close()

def parse_record(example_proto, input_names, output_channels):
    # Parse record from TFRecord file

    all_names = input_names + ["mix"]

    features = {key : tf.io.FixedLenSequenceFeature([], allow_missing=True, dtype=tf.float32) for key in all_names}
    features["length"] = tf.io.FixedLenFeature([], tf.int64)
    features["channels"] = tf.io.FixedLenFeature([], tf.int64)

    parsed_features = tf.io.parse_single_example(example_proto, features)

    # Reshape
    length = tf.cast(parsed_features["length"], tf.int64)
    channels = tf.constant(output_channels, tf.int64) #tf.cast(parsed_features["channels"], tf.int64)
    sample = dict()
    for key in all_names:
        if key is 'mix':
            sample[key] = tf.reshape(parsed_features[key], tf.stack([length, channels]))
        else:
            sample[key] = tf.reshape(parsed_features[key], tf.stack([length, 1]))
            
    sample["length"] = length
    sample["channels"] = channels

    return sample

def get_dataset(model_config, input_shape, output_shape, partition):
    '''
    For a model configuration and input/output shapes of the network, get the corresponding dataset for a given partition
    :param model_config: Model config
    :param input_shape: Input shape of network
    :param output_shape: Output shape of network
    :param partition: "train", "valid", or "test" partition
    :return: Tensorflow dataset object
    '''

    
    # Check if pre-processed dataset is already available for this model config and partition
    dataset_name = "task_" + model_config["task"] + "_" + \
                   "sr_" + str(model_config["expected_sr"]) + "_" + \
                   "mono_" + str(model_config["mono_downmix"])
    main_folder = os.path.join(model_config["data_path"], dataset_name)

    if not os.path.exists(main_folder):
        # We have to prepare the dataset
        print("Preparing dataset! This could take a while...")

        dataset = get_dataset_pickle(model_config)

        # Convert audio files into TFRecords now

        # The dataset structure is a dictionary with "train", "valid", "test" keys, whose entries are lists, where each element represents a song.
        # Each song is represented as a dictionary containing elements mix, acc, vocal or mix, bass, drums, other, vocal depending on the task.

        num_cores = 8

        for curr_partition in ["train", "val", "test"]:
            print("Writing " + curr_partition + " partition...")

            # Shuffle sample order
            sample_list = dataset[curr_partition]
            random.shuffle(sample_list)

            # Create folder
            partition_folder = os.path.join(main_folder, curr_partition)
            os.makedirs(partition_folder)

            part_entries = int(np.ceil(float(len(sample_list) / float(num_cores))))
            processes = list()
            for core in range(num_cores):
                train_filename = os.path.join(partition_folder, str(core) + "_")  # address to save the TFRecords file
                sample_list_subset = sample_list[core * part_entries:min((core + 1) * part_entries, len(sample_list))]
                proc = Process(target=write_records,
                               args=(sample_list_subset, model_config, input_shape, output_shape, train_filename))
                proc.start()
                processes.append(proc)
            for p in processes:
                p.join()

    print("Dataset ready!")
    # Finally, load TFRecords dataset based on the desired partition
    dataset_folder = os.path.join(main_folder, partition)
    records_files = glob.glob(os.path.join(dataset_folder, "*.tfrecords"))
    random.shuffle(records_files)
    dataset = tf.data.TFRecordDataset(records_files)
    dataset = dataset.map(lambda x : parse_record(x, model_config["input_names"], output_shape[-1]), num_parallel_calls=model_config["num_workers"])
    dataset = dataset.prefetch(10)

    # Take random samples from each song
    if partition == "train":
        dataset = dataset.flat_map(lambda x : take_random_snippets(x, model_config["input_names"] + ["mix"], input_shape[1:], output_shape[1:], model_config["num_snippets_per_track"]))
    else:
        dataset = dataset.flat_map(lambda x : take_all_snippets(x, model_config["input_names"] + ["mix"], input_shape[1:], output_shape[1:]))
    dataset = dataset.prefetch(100)

#     if partition == "train" and model_config["augmentation"]: # If its the train partition, activate data augmentation if desired
#             dataset = dataset.map(Utils.random_amplify, num_parallel_calls=model_config["num_workers"]).prefetch(100)

    # Cut outputs to centre part
    dataset = dataset.map(lambda x : Utils.crop_sample(x, (input_shape[1] - output_shape[1])//2)).prefetch(100)

    if partition == "train": # Repeat endlessly and shuffle when training
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=model_config["cache_size"])

    dataset = dataset.batch(model_config["batch_size"], drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset

def get_dataset_pickle(model_config):
    
    if model_config["task"] == "dry":
               
        with open('data/dataDryDict.pkl', "rb") as fp:
            dataset = pickle.load(fp)
        
    elif model_config["task"] == "wet":

        with open('data/dataWetDict.pkl', "rb") as fp:
            dataset = pickle.load(fp)
                
    for partition in ["train", "val", "test"]:

        for idx in range(len(dataset[partition])):

            for key in dataset[partition][idx].keys():

                if dataset[partition][idx][key] is not None:
                    dataset[partition][idx][key] = os.path.join(model_config['enst_path'],dataset[partition][idx][key][1:])


    return dataset     
            
            

def get_path(db_path, instrument_node):
    return db_path + os.path.sep + instrument_node.xpath("./relativeFilepath")[0].text
