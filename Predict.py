from sacred import Experiment
from Config import config_ingredient
import Evaluate
import os

ex = Experiment('Waveunet Prediction', ingredients=[config_ingredient])

@ex.config
def cfg():
    model_path = os.path.join("checkpoints", "baseline_wet", "940358-130000") # Load stereo vocal model by default
#     input_path = {'hi-hat': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/hi-hat/001_hits_snare-drum_sticks_x5.wav',
#  'kick': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/kick/001_hits_snare-drum_sticks_x5.wav',
#  'mix': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/wet_mix/001_hits_snare-drum_sticks_x5_norm.wav',
#  'overhead_L': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/overhead_L/001_hits_snare-drum_sticks_x5.wav',
#  'overhead_R': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/overhead_R/001_hits_snare-drum_sticks_x5.wav',
#  'snare': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/snare/001_hits_snare-drum_sticks_x5.wav',
#  'tom_1': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/tom_1/001_hits_snare-drum_sticks_x5.wav',
#  'tom_2': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/tom_2/001_hits_snare-drum_sticks_x5.wav',
#  'tom_3': '/import/c4dm-04/davem/ENST-drums/drummer_3/audio/hi-hat/001_hits_snare-drum_sticks_x5.wav'}
    
    input_path = {'hi-hat': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/hi-hat/078_phrase_reggae_simple_slow_sticks.wav',
  'kick': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/kick/078_phrase_reggae_simple_slow_sticks.wav',
  'mix': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/wet_mix/078_phrase_reggae_simple_slow_sticks_norm.wav',
  'overhead_L': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/overhead_L/078_phrase_reggae_simple_slow_sticks.wav',
  'overhead_R': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/overhead_R/078_phrase_reggae_simple_slow_sticks.wav',
  'snare': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/snare/078_phrase_reggae_simple_slow_sticks.wav',
  'tom_1': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/tom_1/078_phrase_reggae_simple_slow_sticks.wav',
  'tom_2': '/import/c4dm-04/davem/ENST-drums/drummer_1/audio/tom_2/078_phrase_reggae_simple_slow_sticks.wav',
  'tom_3': None}
    
    output_path = 'audio_examples/mix.wav'
    

@ex.automain
def main(cfg, model_path, input_path, output_path):
    

    model_config = cfg["model_config"]
    
    Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)