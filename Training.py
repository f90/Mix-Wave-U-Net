from sacred import Experiment
from tqdm import tqdm

from Config import config_ingredient
import tensorflow as tf
import numpy as np
import os

import Datasets
import Utils
import Models.MixWaveUNet
import Test

import functools

ex = Experiment('Waveunet Training', ingredients=[config_ingredient])

@ex.config
# Executed for training, sets the seed value to the Sacred config so that Sacred fixes the Python and Numpy RNG to the same state everytime.
def set_seed():
    seed = 1337

@config_ingredient.capture
def train(model_config, experiment_id, load_model=None):
    # Determine input and output shapes
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of input
    if model_config["network"] == "unet":
        model_class = Models.MixWaveUNet.MixWaveUNet(model_config)
    else:
        raise NotImplementedError

    input_shape, output_shape = model_class.get_padding(np.array(disc_input_shape))
    model_func = model_class.get_output

    # Placeholders and input normalisation

    dataset = Datasets.get_dataset(model_config, input_shape, output_shape, partition="train")
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    
    batch_input = tf.concat([batch[key] for key in sorted(batch.keys()) if key != 'mix'], 2)

    print("Training...")

    # BUILD MODELS
    # Separator
    pred_outputs = model_func(batch_input, training=True, reuse=False)

    # Supervised objective: MSE for raw audio, MAE for magnitude space (Jansson U-Net)
    loss = 0
    target_output = batch['mix']
    pred_output = pred_outputs['mix']

    loss += tf.reduce_mean(tf.abs(target_output - pred_output))

    # TRAINING CONTROL VARIABLES
    global_step = tf.compat.v1.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)
    increment_global_step = tf.compat.v1.assign(global_step, global_step + 1)

    # Set up optimizers
    vars = Utils.getTrainableVariables("separator")
    print("Sep_Vars: " + str(Utils.getNumParams(vars)))
    print("Num of variables" + str(len(tf.compat.v1.global_variables())))

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.compat.v1.variable_scope("separator_solver"):
            separator_solver = tf.compat.v1.train.AdamOptimizer(learning_rate=model_config["lr"]).minimize(loss, var_list=vars)

    # SUMMARIES
    tf.compat.v1.summary.scalar("sep_loss", loss, collections=["sup"])
    sup_summaries = tf.compat.v1.summary.merge_all(key='sup')

    # Start session and queue input threads
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter(model_config["log_dir"] + os.path.sep + str(experiment_id),graph=sess.graph)

    # CHECKPOINTING
    # Load pretrained model to continue training, if we are supposed to
    if load_model != None:
        restorer = tf.train.Saver(tf.compat.v1.global_variables(), write_version=tf.compat.v1.train.SaverDef.V2)
        print("Num of variables" + str(len(tf.compat.v1.global_variables())))
        restorer.restore(sess, load_model)
        print('Pre-trained model restored from file ' + load_model)

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), write_version=tf.compat.v1.train.SaverDef.V2)

    # Start training loop
    _global_step = sess.run(global_step)
    _init_step = _global_step
    for _ in tqdm(range(model_config["epoch_it"])):
        # TRAIN SEPARATOR
        #try:
        _, _sup_summaries = sess.run([separator_solver, sup_summaries])
        #except tf.errors.OutOfRangeError as e: # Ignore end of dataset and start over again
        #    continue
        writer.add_summary(_sup_summaries, global_step=_global_step)

        # Increment step counter, check if maximum iterations per epoch is achieved and stop in that case
        _global_step = sess.run(increment_global_step)

    # Epoch finished - Save model
    print("Finished epoch!")
    save_path = saver.save(sess, model_config["model_base_dir"] + os.path.sep + str(experiment_id) + os.path.sep + str(experiment_id), global_step=int(_global_step))

    # Close session, clear computational graph
    writer.flush()
    writer.close()
    sess.close()
    tf.reset_default_graph()

    return save_path

@config_ingredient.capture
def optimise(model_config, experiment_id):
    epoch = 0
    best_loss = 10000
    model_path = None
    best_model_path = None
    for i in range(2):
        worse_epochs = 0
        if i==1:
            print("Finished first round of training, now entering fine-tuning stage")
#             model_config["batch_size"] *= 2
            model_config["lr"] = 1e-5
        while worse_epochs < model_config["worse_epochs"]: # Early stopping on validation set after a few epochs
            print("EPOCH: " + str(epoch))
            model_path = train(load_model=model_path)
            curr_loss = Test.test(model_config, model_folder=str(experiment_id), partition="val", load_model=model_path)
            epoch += 1
            if curr_loss < best_loss:
                worse_epochs = 0
                print("Performance on validation set improved from " + str(best_loss) + " to " + str(curr_loss))
                best_model_path = model_path
                best_loss = curr_loss
            else:
                worse_epochs += 1
                print("Performance on validation set worsened to " + str(curr_loss))
    print("TRAINING FINISHED - TESTING WITH BEST MODEL " + best_model_path)
    test_loss = Test.test(model_config, model_folder=str(experiment_id), partition="test", load_model=best_model_path)
    return best_model_path, test_loss

@ex.automain
def run(cfg):
    model_config = cfg["model_config"]
    print("SCRIPT START")
    # Create subfolders if they do not exist to save results
    for dir in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Optimize in a supervised fashion until validation loss worsens
    sup_model_path, sup_loss = optimise()
    print("Supervised training finished! Saved model at " + sup_model_path + ". Performance: " + str(sup_loss))