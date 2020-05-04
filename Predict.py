from sacred import Experiment
from Config import config_ingredient
import Evaluate
import os

ex = Experiment('Waveunet Prediction', ingredients=[config_ingredient])

@ex.config
def cfg():
    model_path = os.path.join("checkpoints", "wet", "wet-1108000") # Load wet pretrained model by default

    input_path = {'hi-hat': 'audio_examples/inputs/hihat.wav',
  'kick': 'audio_examples/inputs/kick.wav',
  'mix': 'audio_examples/inputs/wet_mix.wav',
  'overhead_L': 'audio_examples/inputs/overheadL.wav',
  'overhead_R': 'audio_examples/inputs/overheadR.wav',
  'snare': 'audio_examples/inputs/snare.wav',
  'tom_1': 'audio_examples/inputs/tom1.wav',
  'tom_2': 'audio_examples/inputs/tom2.wav',
  'tom_3': None}

    output_path = 'audio_examples/outputs/wet_mix.wav'
    

@ex.automain
def main(cfg, model_path, input_path, output_path):
    model_config = cfg["model_config"]
    Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)