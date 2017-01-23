import string
from .nn_column_labeler import NN_Column_Labeler
from .museum_data_reader import Column

# Hyperparameters used by NNetModel:
hp = {}
hp['split_by'] = 'filename'  # name of the column attribute on which to randomly split columns into training and testing sets
hp['subsize'] = 100  # number of row elements (rows) in each bagging subsample
hp['n_samples'] = 1 * 150  # number of subsamples from each column to take when bagging
hp['samples_validation_frac'] = 0.01  # fraction of training samples that are held-out for validation purposes

# ### Hyperparameters for character sequences
hp['maxlen'] = 200  # cut resulting character seqs after this number of chars (ensure all character seq inputs are of the same length)
hp['max_features'] = 128  # number of 'bits' to use when encoding a character (i.e., the length of character vocabulary)

# ### Hyperparameters for character frequencies

hp['char_vocab'] = string.printable  # vocabulary of characters - all printable characters (includes the '\n' character)
hp['entropy'] = True  # whether to add Shannon's entropy to the char_freq feature vectors

# ### Performance metrics for labelers
metrics = ['categorical_accuracy', 'fmeasure', 'MRR']  # list of performance metrics to compare NN column labelers with
metrics_average = 'macro'  # 'macro', 'micro', or 'weighted'


# ### Convolutional NN (CNN) hyperparameters:
hp_cnn = {}
hp_cnn['batch_size'] = 50  # batch training size; a good value is 25, but 50 is faster (while producing similar accuracy)
hp_cnn['dropout'] = 0.5  # dropout value for the dropout layers; no difference between 0.5 and 0.1; reducing below 0.1 seems to slightly hurt the test accuracy (as expected)
hp_cnn['nb_filter'] = 100  # number of filters for the conv layers
hp_cnn['filter_length'] = 3  # 50 # length of the filter window in the conv layer
hp_cnn['border_mode'] = 'valid'  # 'valid' (no zero-padding) or 'same' (with zero padding)
hp_cnn['hidden_dims'] = 100  # number of units for the vanilla (fully connected) hidden layer
hp_cnn['embedding_dims'] = 64  # 128 # dimensionality of character embedding (number of values to squash the initial max_features encoding)
hp_cnn['nb_epoch'] = 10  # 7 # number of training epochs; increasing this beyond 6 does not improve the model
hp_cnn['final_layer_act'] = 'softmax'  # 'linear' # activation function for the last layer
hp_cnn['loss'] = 'categorical_crossentropy'  # 'mse' #'binary_crossentropy' # loss function to use
hp_cnn['metrics'] = metrics
hp_cnn['metrics_average'] = metrics_average
hp_cnn['optimizer'] = 'adam'  # 'rmsprop' # 'adam' # optimization algorithm
hp_cnn['initial_dropout'] = 0.01  # dropout value for the initial layer

# ### Multi-Layer Perceptron (MLP) hyperparameters:
hp_mlp = {}
hp_mlp['batch_size'] = hp_cnn['batch_size']
hp_mlp['pretrain_lr'] = 0.05  # not needed?
hp_mlp['finetune_lr'] = 0.5  # not needed?
hp_mlp['pretraining_epochs'] = 100
hp_mlp['finetuning_epochs'] = 10
hp_mlp['hidden_layers_sizes'] = [100, 100, 100]
hp_mlp['corruption_levels'] = [0.5, 0.0, 0.0]
hp_mlp['activation'] = 'tanh'  # 'tanh' or 'relu' or 'sigmoid'
hp_mlp['final_layer_act'] = 'softmax'  # 'linear' # activation function for the last layer
hp_mlp['loss'] = 'categorical_crossentropy'  # 'mse' #'binary_crossentropy' # loss function to use
hp_mlp['metrics'] = metrics
hp_mlp['metrics_average'] = metrics_average
hp_mlp['optimizer'] = 'adam'  # 'rmsprop' # 'adam' # optimization algorithm