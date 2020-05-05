import tensorflow as tf

def independent_outputs(featuremap, output_names, num_channels, filter_width, padding, activation):
    outputs = dict()
    for name in output_names:
        outputs[name] = tf.layers.conv1d(featuremap, num_channels, filter_width, activation=activation, padding=padding)
    return outputs