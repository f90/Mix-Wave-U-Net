import soundfile
from sacred import Experiment
from Config import config_ingredient
import Evaluate, Datasets, Utils
import os
import numpy as np
import librosa

ex = Experiment('Waveunet Prediction', ingredients=[config_ingredient])

@ex.config
def cfg():
    model_path = os.path.join("checkpoints", "75631", "75631-150000") # Load stereo vocal model by default
#     input_path = {'hi-hat': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/hi-hat/001_hits_snare-drum_sticks_x5.wav',
#  'kick': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/kick/001_hits_snare-drum_sticks_x5.wav',
#  'mix': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/wet_mix/001_hits_snare-drum_sticks_x5_norm.wav',
#  'overhead_L': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/overhead_L/001_hits_snare-drum_sticks_x5.wav',
#  'overhead_R': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/overhead_R/001_hits_snare-drum_sticks_x5.wav',
#  'snare': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/snare/001_hits_snare-drum_sticks_x5.wav',
#  'tom_1': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/tom_1/001_hits_snare-drum_sticks_x5.wav',
#  'tom_2': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/tom_2/001_hits_snare-drum_sticks_x5.wav',
#  'tom_3': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/hi-hat/001_hits_snare-drum_sticks_x5.wav'}
    
#     input_path = {'hi-hat': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/hi-hat/078_phrase_reggae_simple_slow_sticks.wav',
#   'kick': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/kick/078_phrase_reggae_simple_slow_sticks.wav',
#   'mix': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/wet_mix/078_phrase_reggae_simple_slow_sticks_norm.wav',
#   'overhead_L': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/overhead_L/078_phrase_reggae_simple_slow_sticks.wav',
#   'overhead_R': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/overhead_R/078_phrase_reggae_simple_slow_sticks.wav',
#   'snare': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/snare/078_phrase_reggae_simple_slow_sticks.wav',
#   'tom_1': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/tom_1/078_phrase_reggae_simple_slow_sticks.wav',
#   'tom_2': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/tom_2/078_phrase_reggae_simple_slow_sticks.wav',
#   'tom_3': None}

#     
    output_path = 'audio_examples/'
    

@ex.automain
def main(cfg, model_path, output_path):
    

    model_config = cfg["model_config"]
    
    dataset = Datasets.get_dataset_pickle(model_config)
    
    L1 = []
    L2 = []
    for track in dataset['test']:
        
        output_track = os.path.basename(track['mix'])  
        output_track = os.path.join(output_path,output_track)
        
        print(output_track)
        
        Evaluate.produce_outputs(model_config, model_path, track, output_track)
        
        target, sr = Utils.load(track['mix'], sr=None, mono=False) 
        target = target/np.max(np.abs(target))
        
        soundfile.write(output_track+'_target.wav', target, sr)
        
        output, _ = Utils.load(output_track, sr=None, mono=False) 
        
        l1 = np.mean(np.abs(target-output))
        l2 = np.mean(np.square(target-output))
        
        L1.append(l1)
        L2.append(l2)
    
    print('L1: %.8f' % np.mean(np.asarray(L1)))
    print('L2: %.8f' % np.mean(np.asarray(L2)))
        
        
        