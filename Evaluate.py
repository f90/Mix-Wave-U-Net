import numpy as np
import soundfile
import tensorflow as tf

import os

import Models.MixWaveUNet
import Utils

def predict(audio, model_config, load_model):
    # Determine input and output shapes, if we use U-net as separator
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of discriminator input
    if model_config["network"] == "unet":
        model_class = Models.MixWaveUNet.MixWaveUNet(model_config)
    else:
        raise NotImplementedError

    input_shape, output_shape = model_class.get_padding(np.array(disc_input_shape))
    model_func = model_class.get_output

    # Batch size of 1
    input_shape[0] = 1
    output_shape[0] = 1

    tracks_ph = tf.compat.v1.placeholder(tf.float32, input_shape)

    print("Testing...")

    # BUILD MODELS
    # Separator
    frame_pred = model_func(tracks_ph, training=False, reuse=False)

    # Start session and queue input threads
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Load model
    # Load pretrained model to continue training, if we are supposed to
    restorer = tf.compat.v1.train.Saver(None, write_version=tf.compat.v1.train.SaverDef.V2)
    print("Num of variables" + str(len(tf.compat.v1.global_variables())))
    restorer.restore(sess, load_model)
    print('Pre-trained model restored for song prediction')

    mix_pred = predict_track(model_config, sess, audio, input_shape, output_shape, frame_pred, tracks_ph)

    # Close session, clear computational graph
    sess.close()
    tf.compat.v1.reset_default_graph()

    return mix_pred

def predict_track(model_config, sess, audio, input_shape, output_shape, frame_pred, tracks_ph):
    '''
    Outputs estimates given Tensorflow session and placeholders belonging to the prediction network.
    :param model_config: Model configuration dictionary
    :param sess: Tensorflow session used to run the network inference
    :param audio: Collection of audio signals
    :param input_shape: Input shape of model ([batch_size, num_samples, num_channels])
    :param output_shape: Output shape of model ([batch_size, num_samples, num_channels])
    :param frame_pred: List of Tensorflow tensors that represent the output of the network
    :param tracks_ph: Input tensor of the network
    :return:
    '''
    
    for key in audio.keys():
        
        # Append zeros to mixture if its shorter than input size of network - this will be cut off at the end again
        if audio[key].shape[0] < input_shape[1]:
            extra_pad = input_shape[1] - audio[key].shape[0]
            audio[key] = np.pad(audio[key], [(0, extra_pad), (0,0)], mode="constant", constant_values=0.0)
        else:
            extra_pad = 0

    # Preallocate source predictions (same shape as input mixture)
    mix_time_frames = audio[key].shape[0]
    mix_preds = np.asfortranarray(np.zeros((mix_time_frames, output_shape[2]), np.float32))

    input_time_frames = input_shape[1]
    output_time_frames = output_shape[1]

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
            
#         audio_tensor = audio for name in model_config["input_names"]}
        audio_part = np.concatenate([audio_part[key] for key in sorted(audio_part.keys())], axis=2)
        
        mix_part = sess.run(frame_pred, feed_dict={tracks_ph: audio_part})
      
        # Save predictions
        # source_shape = [1, freq_bins, acc_mag_part.shape[2], num_chan]
        mix_preds[mix_pos:mix_pos + output_time_frames] = mix_part['mix'][0, :, :]

    # In case we had to pad the mixture at the end, remove those samples from source prediction now
    if extra_pad > 0:
        mix_preds = mix_preds[:-extra_pad,:] 

    return mix_preds

def produce_outputs(model_config, load_model, tracksdict, output_path):
    '''
    For a given input, save outputs made by a given model.
    :param model_config: Model configuration
    :param load_model: Model checkpoint path
    :param tracksdict: Dictionary containing information about input files
    :param output_path: Output directory where estimated sources should be saved. Defaults to the same folder as the input file, if not given
    :return: Dictionary of estimates
    '''
    print("Producing outputs for input files " + str(tracksdict))
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
               

    mix_pred = predict(audio, model_config, load_model)
    
    # Save source estimates as audio files into output dictionary
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        print("WARNING: Given output path " + directory + " does not exist. Trying to create it...")
        os.makedirs(directory)
    assert(os.path.exists(directory))
    print(output_path)
    soundfile.write(output_path, mix_pred, sr)

