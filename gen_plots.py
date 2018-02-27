import numpy as np
import matplotlib.pyplot as plt
import itertools


#Adapted from Canvas
def write_heatmap(in_file, out_file):
    conf_matrix = np.load(in_file).astype('int')
    plt.imshow(conf_matrix, interpolation='nearest')
    plt.colorbar()

    thresh = conf_matrix.max() / 2. # threshold for printing the numbers in black or white
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="black" if conf_matrix[i, j] > thresh else "white")

    plt.tight_layout()
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_file)
    plt.show()
    plt.close() # Closes for the next plot

def write_plot(train_file, validation_file, out_file):
    train_vals = np.load(train_file)
    validation_vals = np.load(validation_file)
    num_epochs = len(train_vals)
    x = np.linspace(0, num_epochs, num_epochs)
    plt.plot(x, train_vals, 'b-', label='Training')
    plt.plot(x, validation_vals, 'r-', label='Validation')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epochs')
    plt.savefig(out_file, bboxinches = 'tight')
    plt.show()
    plt.close()
