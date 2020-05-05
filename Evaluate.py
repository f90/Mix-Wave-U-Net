import numpy as np
import soundfile
import tensorflow as tf

import os

import Models.UnetAudioSeparator
import Utils

def predict(audio, model_config, load_model):
    '''
    Function in accordance with MUSB evaluation API. Takes MUSDB track object and computes corresponding source estimates, as well as calls evlauation script.
    Model has to be saved beforehand into a pickle file containing model configuration dictionary and checkpoint path!
    :param audio: Track object
    :return: Source estimates dictionary
    '''

    # Determine input and output shapes, if we use U-net as separator
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of discriminator input
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config)
    else:
        raise NotImplementedError

    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    # Batch size of 1
    sep_input_shape[0] = 1
    sep_output_shape[0] = 1

    tracks_ph = tf.compat.v1.placeholder(tf.float32, sep_input_shape)

    print("Testing...")

    # BUILD MODELS
    # Separator
    frame_pred = separator_func(tracks_ph, training=False, reuse=False)

    # Start session and queue input threads
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Load model
    # Load pretrained model to continue training, if we are supposed to
    restorer = tf.compat.v1.train.Saver(None, write_version=tf.compat.v1.train.SaverDef.V2)
    print("Num of variables" + str(len(tf.compat.v1.global_variables())))
    restorer.restore(sess, load_model)
    print('Pre-trained model restored for song prediction')

#     mix_audio, orig_sr, mix_channels = track.audio, track.rate, track.audio.shape[1] # Audio has (n_samples, n_channels) shape
    mix_pred = predict_track(model_config, sess, audio, sep_input_shape, sep_output_shape, frame_pred, tracks_ph)

    # Upsample predicted source audio and convert to stereo. Make sure to resample back to the exact number of samples in the original input (with fractional orig_sr/new_sr this causes issues otherwise)
#     pred_audio =Utils.resample(mix_pred, model_config["expected_sr"], orig_sr)[:mix_audio.shape[0],:] 

    # Close session, clear computational graph
    sess.close()
    tf.compat.v1.reset_default_graph()

    return mix_pred

def predict_track(model_config, sess, audio, sep_input_shape, sep_output_shape, frame_pred, tracks_ph):
    '''
    Outputs source estimates for a given input mixture signal mix_audio [n_frames, n_channels] and a given Tensorflow session and placeholders belonging to the prediction network.
    It iterates through the track, collecting segment-wise predictions to form the output.
    :param model_config: Model configuration dictionary
    :param sess: Tensorflow session used to run the network inference
    :param mix_audio: [n_frames, n_channels] audio signal (numpy array). Can have higher sampling rate or channels than the model supports, will be downsampled correspondingly.
    :param sep_input_shape: Input shape of separator ([batch_size, num_samples, num_channels])
    :param sep_output_shape: Input shape of separator ([batch_size, num_samples, num_channels])
    :param frame_pred: List of Tensorflow tensors that represent the output of the separator network
    :param tracks_ph: Input tensor of the network
    :return:
    '''


    
    for key in audio.keys():
        
        # Append zeros to mixture if its shorter than input size of network - this will be cut off at the end again
        if audio[key].shape[0] < sep_input_shape[1]:
            extra_pad = sep_input_shape[1] - audio[key].shape[0]
            audio[key] = np.pad(audio[key], [(0, extra_pad), (0,0)], mode="constant", constant_values=0.0)
        else:
            extra_pad = 0

    # Preallocate source predictions (same shape as input mixture)
    mix_time_frames = audio[key].shape[0]
    mix_preds = np.asfortranarray(np.zeros((mix_time_frames, sep_output_shape[2]), np.float32))

    input_time_frames = sep_input_shape[1]
    output_time_frames = sep_output_shape[1]

    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_time_frames = (input_time_frames - output_time_frames) // 2
    

    for key in audio.keys():     
        audio[key] = np.pad(audio[key], [(pad_time_frames, pad_time_frames), (0,0)], mode="constant", constant_values=0.0)

    # Iterate over mixture magnitudes, fetch network rpediction
    for mix_pos in range(0, mix_time_frames, output_time_frames):
        # If this output patch would reach over the end of the source spectrogram, set it so we predict the very end of the output, then stop
        if mix_pos + output_time_frames > mix_time_frames:
            mix_pos = mix_time_frames - output_time_frames
        
        audio_part = {}
        for key in audio.keys():     
        # Prepare mixture excerpt by selecting time interval
            audio_part[key] = audio[key][mix_pos:mix_pos + input_time_frames,:]
            audio_part[key] = np.expand_dims(audio_part[key], axis=0)
            
#         audio_tensor = audio for name in model_config["source_names"]}   
        audio_part = np.concatenate([audio_part[key] for key in sorted(audio_part.keys())], axis=2)
        
        mix_part = sess.run(frame_pred, feed_dict={tracks_ph: audio_part})
      
        # Save predictions
        # source_shape = [1, freq_bins, acc_mag_part.shape[2], num_chan]
        mix_preds[mix_pos:mix_pos + output_time_frames] = mix_part['mix'][0, :, :]

    # In case we had to pad the mixture at the end, remove those samples from source prediction now
    if extra_pad > 0:
        mix_preds = mix_preds[:-extra_pad,:] 

    return mix_preds




def produce_source_estimates(model_config, load_model, tracksdict, output_path):
    '''
    For a given input mixture file, saves source predictions made by a given model.
    :param model_config: Model configuration
    :param load_model: Model checkpoint path
    :param input_path: Path to input mixture audio file
    :param output_path: Output directory where estimated sources should be saved. Defaults to the same folder as the input file, if not given
    :return: Dictionary of source estimates containing the source signals as numpy arrays
    '''
    print("Producing source estimates for input mixture file " + str(tracksdict))
    # Prepare input audio as track object (in the MUSDB sense), so we can use the MUSDB-compatible prediction function
    audio = {}
    lengths = []
    for key, value in tracksdict.items():
        
        if value is not None:
            if key != 'mix':      
                
                audioTrack, sr = Utils.load(value, sr=model_config["expected_sr"], mono=model_config["mono_downmix"])
                assert (audioTrack.shape[1]==1)
                assert (sr==model_config['expected_sr'])
                audio[key] = audioTrack
                lengths.append(audioTrack.shape[0])
    length = list(set(lengths))            
    assert(len(length)==1)
                            
    for key, value in tracksdict.items():
        
        if value is None:
            if key != 'mix': 
                audio[key] = np.zeros((length[0],1), dtype=np.float32)     
               

    mix_pred = predict(audio, model_config, load_model) # Input track to prediction function, get source estimates
    
    # Save source estimates as audio files into output dictionary
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        print("WARNING: Given output path " + directory + " does not exist. Trying to create it...")
        os.makedirs(directory)
    assert(os.path.exists(directory))
    print(output_path)
    soundfile.write(output_path, mix_pred, sr)

