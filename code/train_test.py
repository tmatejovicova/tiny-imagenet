import os
import load_data
import models
import plot_history
import constants
import pickle
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import shutil


def train_test(model, model_name, batch_size, augment, epochs):
    # Model summary
    print("Training model " + model_name)
    model.summary()

    #  Generate label dictionaries
    label_list, label_dict, class_dict = load_data.build_label_dicts()
    # Obtain generators
    train_generator, val_generator = load_data.get_generators(label_list, label_dict, class_dict, batch_size, augment)

    # Train
    results_dir = constants.RESULTS_DIR + model_name + '/'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    history = train_model(model, train_generator, batch_size, epochs, val_generator, results_dir)
    plot_history.plot_history(history, results_dir)
    model.save_weights(results_dir + 'weights.h5')

    # Test model
    test_model(model, val_generator, results_dir)

    print()

def test_saved(model, model_name, batch_size, augment):
    # Load weights
    results_dir = constants.RESULTS_DIR + model_name + '/'
    model.load_weights(results_dir + 'weights.h5')

    #  Generate label dictionaries
    label_list, label_dict, class_dict = load_data.build_label_dicts()
    # Obtain generators
    _, val_generator = load_data.get_generators(label_list, label_dict, class_dict, batch_size, augment)

    # Test
    test_model(model, val_generator, results_dir)

def train_model(model, train_generator, batch_size, epochs, val_generator, dir):
    # tb_callback = TensorBoard(log_dir = './log/run', update_freq = 'batch')
    # callbacks = [tb_callback]

    weights_dir = dir + 'weights/'
    if os.path.exists(weights_dir):
        shutil.rmtree(weights_dir)

    os.makedirs(weights_dir)

    filepath = weights_dir + 'weights.{epoch:02d}-{val_acc:.2f}.h5'
    mc_callback = ModelCheckpoint(filepath, monitor = 'val_acc', save_weights_only = True, period = 1)
    callbacks = [mc_callback]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch = constants.NUM_TRAIN_SAMPLES // batch_size,
        epochs = epochs,
        validation_data = val_generator,
        validation_steps = constants.NUM_VAL_SAMPLES // batch_size,
        verbose = 2,
        callbacks = callbacks)
    with open(dir + 'train_history.pickle', 'wb') as f:
        pickle.dump(history, f)
    print('History\n' + str(history.history))
    with open(dir + 'train_history.txt', 'wb') as f:
        pickle.dump(str(history.history), f)
    return history

def test_model(model, val_generator, dir):
    score = model.evaluate_generator(
        val_generator,
        steps = constants.NUM_VAL_SAMPLES,
        verbose = 2)
    print(str(model.metrics_names))
    print('Val loss: ' + str(score[0]))
    print('Val top-1 accuracy: ' + str(score[1]))
    print('Val top-5 accuracy: ' + str(score[2]))

    with open(dir + 'test_result.txt', 'w') as f:
        f.write(str(model.metrics_names))
        f.write('\nVal loss: ' + str(score[0]))
        f.write('\nVal top-1 accuracy: ' + str(score[1]))
        f.write('\nVal top-5 accuracy: ' + str(score[2]))
