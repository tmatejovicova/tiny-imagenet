import matplotlib.pyplot as plt
import constants
import pickle

def plot_history_from_file(model_name):
    dir = constants.RESULTS_DIR + model_name + '/'
    with open(dir + 'train_history.pickle', 'rb') as f:
        history = pickle.load(f)
    plot_history(history, dir)

def plot_history(history, dir):
    # Plot training & validation top-1 accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model top-1 accuracy')
    plt.ylabel('Top-1 accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.savefig(dir + 'top1accuracy.jpg')
    plt.close()

    # Plot training & validation top-5 accuracy values
    plt.plot(history.history['top_k_categorical_accuracy'])
    plt.plot(history.history['val_top_k_categorical_accuracy'])
    plt.title('Model top-5 accuracy')
    plt.ylabel('Top-5 accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.savefig(dir + 'top5accuracy.jpg')
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.savefig(dir + 'loss.jpg')
    plt.close()

    print(history.history)
