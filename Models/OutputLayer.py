import tensorflow as tf

import Utils

def independent_outputs(featuremap, source_names, num_channels, filter_width, padding, activation):
    outputs = dict()
    for name in source_names:
        outputs[name] = tf.layers.conv1d(featuremap, num_channels, filter_width, activation=activation, padding=padding)
#         tf.summary.audio(name, outputs[name], 44100, max_outputs=1, collections=["sup"])
    return outputs

def difference_output(input_mix, featuremap, source_names, num_channels, filter_width, padding, activation, training):
    outputs = dict()
    sum_source = 0
    for name in source_names[:-1]:
        out = tf.layers.conv1d(featuremap, num_channels, filter_width, activation=activation, padding=padding)
        outputs[name] = out
        sum_source = sum_source + out

    # Compute last source based on the others
    last_source = Utils.crop(input_mix, sum_source.get_shape().as_list()) - sum_source
    last_source = Utils.AudioClip(last_source, training)
    outputs[source_names[-1]] = last_source
    return outputs