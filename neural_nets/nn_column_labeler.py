
import pandas as pd
from pandas import ExcelWriter
import itertools as it
import os
import os.path
import numpy as np
import string
import random
import copy
import re
import shutil

import subprocess
from subprocess import STDOUT, PIPE

from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

# A hack to fix the AttributeError: module 'tensorflow.python' has no attribute 'control_flow_ops':
# https://github.com/fchollet/keras/issues/3857 :
import tensorflow as tf

tf.python.control_flow_ops = tf

from collections import OrderedDict
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, Convolution1D, MaxPooling1D, Activation, Flatten
from keras.utils.np_utils import to_categorical
import keras.metrics

from .museum_data_reader import Reader
museum_reader = Reader()

# ## Hyperparameters
# ### Data sampling hyperparameters:
VERBOSE = 0

hp = {}
hp['split_by'] = 'filename'  # name of the column attribute on which to randomly split columns into training and testing sets
# 'id' for splitting by column attribute (title@filename)
# 'filename' for splitting by data source filename
hp['cols_test_frac'] = 0.0  # fraction of all data columns that are used for testing (the rest is for training)
hp['subsize'] = 100  # number of row elements (rows) in each bagging subsample
hp['n_samples'] = 1 * 150  # number of subsamples from each column to take when bagging
hp['samples_validation_frac'] = 0.01  # fraction of training samples that are held-out for validation purposes

# ### Hyperparameters for character sequences

hp['maxlen'] = 250 #200  # cut resulting character seqs after this number of chars (ensure all character seq inputs are of the same length)
hp['max_features'] = 128  # number of 'bits' to use when encoding a character (size of character vocabulary)

# ### Hyperparameters for character frequencies

hp['char_vocab'] = string.printable  # vocabulary of characters - all printable characters (includes the '\n' character)
hp['entropy'] = True  # whether to add Shannon's entropy to the char_freq feature vectors

# ### Performance metrics for labelers

metrics = ['categorical_accuracy', 'fmeasure', 'MRR']  # list of performance metrics to compare column labelers with
metrics_average = 'macro'  # 'macro', 'micro', or 'weighted'

# ### Convolutional NN (CNN) hyperparameters:

hp_cnn = {}
hp_cnn['batch_size'] = 50  # batch training size; a good value is 25, but 50 is faster (while producing similar accuracy)
hp_cnn['dropout'] = 0.5  # dropout value for the dropout layers; no difference between 0.5 and 0.1; reducing below 0.1 seems to slightly hurt the test accuracy (as expected)
hp_cnn['n_conv_layers'] = 2 # number of 1d convolutional layers in CNN model
hp_cnn['nb_filter'] = 100  # number of filters for the conv layers
hp_cnn['filter_length'] = 3  # 50 # length of the filter window in the conv layer
hp_cnn['border_mode'] = 'valid'  # 'valid' (no zero-padding) or 'same' (with zero padding)
hp_cnn['hidden_dims'] = 100  # number of units for the vanilla (fully connected) hidden layer
hp_cnn['embedding_dims'] = 64  # 128 # dimensionality of character embedding (number of values to squash the initial max_features encoding)
hp_cnn['nb_epoch'] = 10 # number of training epochs; increasing this beyond 6 does not improve the model
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

class CNN(object):
    """CNN model class"""

    def __init__(self, hp):
        self.hp = hp
        self.model = Sequential()
        self.metrics = [m for m in hp['metrics'] if m in keras.metrics.__dict__.keys()]

    def build(self, n_classes):
        """build the cnn model"""

        self.model.add(Embedding(  # embed the input vectors of dim=max_features to dense vectors of dim=embedding_dims
            input_dim=self.hp['max_features'],
            # note that the embedding matrix W is learned (along with the weights of the other layers) when the whole model is trained, and hopefully the learned embedding matrix W is a "good" one
            output_dim=self.hp['embedding_dims'],
            input_length=self.hp['maxlen'],
            dropout=self.hp['initial_dropout']
        ))

        # we add a Convolution1D, which will learn nb_filter
        # character group filters of size filter_length:
        for l in range(self.hp['n_conv_layers']):
            self.model.add(Convolution1D(nb_filter=self.hp['nb_filter'],
                                         filter_length=self.hp['filter_length'],
                                         border_mode=self.hp['border_mode'],
                                         activation='relu',
                                         subsample_length=1))
            # model.add(Dropout(hp['dropout']))

        # add max pooling:
        self.model.add(MaxPooling1D(pool_length=self.model.output_shape[1]))


        # Flatten the output of the top conv block,
        # so that we can add a vanilla dense layer:
        self.model.add(Flatten())
        self.model.add(Dropout(self.hp['dropout']))

        # add a fully connected hidden layer:
        self.model.add(Dense(self.hp['hidden_dims'], activation='relu'))
        #         self.model.add(Dropout(self.hp['dropout']))

        # Finally, add the final fully connected layer to perform classification:
        self.model.add(Dense(
            n_classes,
            activation=self.hp['final_layer_act']))  # ,
        # W_regularizer=l2(0.01),
        # activity_regularizer=activity_l2(0.01))) #activation='sigmoid'))

        self.model.compile(
            optimizer=self.hp['optimizer'],
            loss=self.hp['loss'],
            metrics=self.metrics)

    def summary(self):
        print("\nModel structure:")
        print(self.model.summary())
        print()

    def train(self, X_train, y_train, X_valid, y_valid):
        print('Training the cnn model...')
        self.model.fit(X_train, y_train,
                       batch_size=self.hp['batch_size'],
                       nb_epoch=self.hp['nb_epoch'],
                       validation_data=[X_valid, y_valid], verbose=VERBOSE)
        print('Training is complete.')

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, batch_size=self.hp['batch_size'])

    def save(self, filename='cnn_model.h5'):
        self.model.save(filename)


class CNN_embedder(object):
    """CNN embedder model class: a model that uses the weights of a CNN-class model's layers up to the Flatten layer"""

    def __init__(self, cnn):
        self.cnn = cnn
        self.hp = cnn.hp  # hyperparameters of the cnn model that was passed as an argument

        """build the cnn embedder model"""
        self.model = Sequential()

        self.model.add(Embedding(  # embed the input vectors of dim=max_features to dense vectors of dim=embedding_dims
            self.hp['max_features'],
            # note that the embedding matrix W has been learned (along with the weights of other layers) when the whole CNN model was trained, and hopefully the learned embedding matrix W is a "good" one
            self.hp['embedding_dims'],
            input_length=self.hp['maxlen'],
            dropout=0,  # dropout is not needed, since we are not learning the weights
            weights=self.cnn.model.layers[0].get_weights()
            # initialize the weights to those of layers[0] (the Embedding layer) of trained cnn model
        ))

        # we add a Convolution1D, which has learned nb_filter
        # character group filters of size filter_length:
        for l in range(self.hp['n_conv_layers']):
            self.model.add(Convolution1D(nb_filter=self.hp['nb_filter'],
                                         filter_length=self.hp['filter_length'],
                                         border_mode=self.hp['border_mode'],
                                         activation='relu',
                                         subsample_length=1,
                                         weights=self.cnn.model.layers['convolution1d_'+str(l+1)].get_weights()
                                         ))
        # model.add(Dropout(hp['dropout']))
        # add max pooling:
        self.model.add(MaxPooling1D(pool_length=self.model.output_shape[1]))

        # Flatten the output of the conv layer to get the output of the CNN_embedder (to be used as input for, e.g., a RF model)
        self.model.add(Flatten())

    def summary(self):
        print("\nModel structure:")
        print(self.model.summary())
        print()

    def predict(self, X):
        return self.model.predict(X)


class MLP(object):
    """Multi-Layer Perceptron with character frequencies+entropy as input"""

    def __init__(self, hp, pretraining=False):
        self.hp = hp
        self.pretraining = pretraining
        self.model = Sequential()
        self.metrics = [m for m in hp['metrics'] if m in keras.metrics.__dict__.keys()]

    def build(self, input_dim, n_classes):
        if not self.pretraining:
            '''no pretraining'''
            for i, (units, dropout) in enumerate(zip(self.hp['hidden_layers_sizes'], self.hp['corruption_levels']),
                                                 start=1):
                activation = self.hp['activation']
                print('adding layer', i, 'with', units, 'units, dropout =', dropout, ', and', activation, 'activation')
                self.model.add(Dense(units, input_shape=(input_dim,)))
                self.model.add(Activation(activation))
                self.model.add(Dropout(dropout))
            print('adding the final layer on top with', self.hp['final_layer_act'], 'activation...')
            self.model.add(Dense(n_classes, activation=self.hp['final_layer_act']))

            print('Compiling the model...')
            self.model.compile(
                optimizer=self.hp['optimizer'],
                loss=self.hp['loss'],
                metrics=self.metrics)

        else:
            '''pretraining+finetuning'''
            print('Pretraining not yet implemented')

    def summary(self):
        print("\nModel structure:")
        print(self.model.summary())
        print()

    def train(self, X_train, y_train, X_valid, y_valid):
        print('Training the mlp model...')
        self.model.fit(X_train, y_train,
                       batch_size=self.hp['batch_size'],
                       nb_epoch=self.hp['finetuning_epochs'],
                       validation_data=[X_valid, y_valid], verbose=VERBOSE)
        print('Training is complete.')

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, batch_size=self.hp['batch_size'])

    def save(self, filename='mlp_model.h5'):
        self.model.save(filename)


def char_freq(X, freq=True, lowercase=True, entropy=True):
    '''extract character counts/frequencies in X (X is a list of ord(character) sequences (list of lists))'''

    all_chars = hp['char_vocab']
    X_charfreq = []
    for i in range(len(X)):
        text = ''.join([chr(code) for code in (X[i])])
        text = str(text)  # concatenate the elements of text into a single string

        if lowercase:
            text = text.lower()
        char_dic = {}
        if text is not '':
            for x in all_chars:
                char_dic[x] = float(text.count(x)) / len(text) if freq else text.count(x)
        else:
            for x in all_chars:
                char_dic[x] = 0

        # add the information measure (negative entropy) of text, Sum(p(i) log(p(i)), i=1,...,n), where p(i) is the i-th character frequency,
        # n is the number of possible characters in the "alphabet", i.e., number of possible states

        entr = 0
        max_entr = -np.log2(1. / len(all_chars))  # maximum entropy, in bits per character, for a text made of all_chars
        if entropy:
            for x in all_chars:
                p_x = char_dic[x]
                if p_x > 0:
                    entr += - p_x * np.log2(p_x)  # entropy of text, in bits per character

        char_dic['_entropy_'] = entr / max_entr  # return the normalized entropy of text
        X_charfreq.append(char_dic)

    X_charfreq = np.array(pd.DataFrame(X_charfreq))
    return X_charfreq


def zero_one_errors(classifier, x, y):
    """Return a float representing the number of errors in the minibatch
    over the total number of examples of the minibatch ; zero one
    loss over the size of the minibatch

    classifier: classifier to use for prediction
    x: array of feature vectors to predict for (new data)
    y: array of true labels

    """

    if type(classifier) == CNN or type(classifier) == MLP:
        y_pred = classifier.predict(x)
        y_pred = np.array([np.argmax(y_pred[i,]) for i in range(y_pred.shape[0])])
    else:
        y_pred = classifier.predict(x)

    # check if y has same dimension as y_pred
    if y.ndim != y_pred.ndim:
        raise TypeError(
            'y should have the same shape as y_pred',
            ('y', y.shape, 'y_pred', y_pred.shape)
        )
    return np.mean(y_pred != y)


def diff(first, second):
    '''difference between two lists'''
    second = set(second)
    return [item for item in first if item not in second]


def train_validation_split(X, y, valid_frac):
    '''Split X, y into (X_train, y_train) and (X_valid, y_valid) according to valid_frac'''
    ind_all = range(len(X))  # indices of X
    ind_valid = random.sample(ind_all, int(
        np.ceil(len(ind_all) * valid_frac)))  # indices of X, y to be assigned to X_valid, y_vald
    ind_train = diff(ind_all, ind_valid)  # indices of X, y to be assigned to X_train, y_train
    X_train = [X[i] for i in ind_train]
    y_train = [y[i] for i in ind_train]
    X_valid = [X[i] for i in ind_valid]
    y_valid = [y[i] for i in ind_valid]
    return (X_train, y_train), (X_valid, y_valid)


class NN_Column_Labeler(object):
    """Predicts labels of Column objects passed as argument, using 'elementary' classifiers specified as a list of names"""

    def _char_filter(self, X, max_ord):
        """Remove characters with ord>max_ord from elements of X"""
        X_filtered = []
        for x in X:
            x = [i for i in x if i <= max_ord]
            X_filtered.append(x)

        return X_filtered

    def _generate_features(self, X, suffix='', verbose=False):
        """Extract features from X['raw']"""

        if any(f in list(it.chain.from_iterable([t.split('@')[-1].split('_') for t in self.classifier_types]))
               for f in ['charseq', 'augmented']):
            if verbose: print("Padding character sequences in X" + suffix + "[\'raw\'] to maxlen =", hp['maxlen'],
                              "...")
            X['charseq'] = sequence.pad_sequences(self._char_filter(X['raw'],hp['max_features']), maxlen=hp['maxlen'], truncating='post')  # Before padding, remove characters beyond hp['max_features'] from X['raw']:

        if any(f in list(it.chain.from_iterable([t.split('@')[-1].split('_') for t in self.classifier_types]))
               for f in ['charfreq', 'augmented']):
            # prepare the character frequencies data for the models that take character frequencies as input
            if verbose: print("Calculating character frequencies in X" + suffix + "[\'raw\'] sequences...")
            X['charfreq'] = char_freq(X['raw'], freq=True, lowercase=True, entropy=hp['entropy'])

        if any(f in [t.split('@')[-1] for t in self.classifier_types]
               for f in ['charseq_embedded', 'augmented']):
            try:
                X['charseq_embedded'] = self.cnn_embedder.predict(X['charseq'])
            except:
                X['charseq_embedded'] = None

        if 'augmented' in [t.split('@')[-1] for t in self.classifier_types]:
            # augment X_charseq_embedded and X_charfreq:
            try:
                X['augmented'] = np.concatenate((X['charseq_embedded'], X['charfreq']), axis=1)
            except:
                X['augmented'] = None
                # X['augmented'] = np.concatenate((X['charseq'], X['charfreq']), axis=1)

    def _predict_soft(self, classifier, X):
        """Predict average soft labels (class probabilities) of X by averaging predictions of classifier for all rows of X"""
        n_classes = len(self.labels)
        if type(classifier) == CNN or type(classifier) == MLP:
            pred_proba = classifier.predict(X)  # this is the output of the top (softmax) layer
        else:  # classifier is a RF
            pred_all = np.zeros([X.shape[0],
                                 n_classes])  # initialise class probabilities matrix with columns corresponding to ALL possible classes
            pred = classifier.predict_proba(X)  # !!! columns in pred correspond to classes in the training set only!
            classes = classifier.classes_  # classes known to the classifier (classes in the training set)
            # now we need to map pred into pred_all:
            pred_all[:,
            classes] = pred  # fill the columns corresponding to classes known to the classifier with classifier's predictions, leaving the rest of columns as 0
            pred_proba = pred_all  # predict class probabilities

        pred_proba_mean = np.mean(pred_proba, axis=0)
        return pred_proba_mean

    def _predict_hard(self, y_score_mean, thresholds=None):
        """Evaluate the hard label of the column, based on the thresholds:"""
        if thresholds is None:
            thresholds = np.repeat(1. / len(self.labels), len(self.labels))  # default thresholds = equal thresholds for all classes
        else:
            thresholds = np.array([v for k, v in thresholds.items()])
        y_pred = np.argmax(y_score_mean - thresholds)
        label_pred = self.inverted_lookup[y_pred]  # choose the label whose threshold-offset score is the largest

        return y_pred, label_pred

    def __init__(self, classifier_types, all_cols, split_by, test_frac, add_headers, p_header):
        """Arguments specify which classifier types to use, and a list of columns to use for training and testing
        add_headers argument specifies whether to add column headers in front of the bagged samples from columns"""
        self.classifier_types = classifier_types
        self.all_cols = all_cols
        self.add_headers = add_headers
        self.p_header = p_header

        self.X_train = {}
        self.X_valid = {}
        self.X_test = {}

        # Initialize a dict of classifiers to be trained:
        self.classifiers = {}
        for t in self.classifier_types:
            self.classifiers[t] = None

        # Randomly split all_cols on split_by attribute into train and test cols, and sample from train and test cols:
        print('Randomly splitting columns on', split_by, '...')
        attributes = list(np.unique([c.__dict__[split_by] for c in all_cols]))
        attributes_test = random.sample(attributes, int(
            np.ceil(len(attributes) * test_frac)))  # leave test_fract (randomly selected) data sources out for testing
        attributes_train = diff(attributes, attributes_test)  # use the rest for training

        self.train_cols = [c for c in all_cols if c.__dict__[split_by] in attributes_train]  # train cols
        self.test_cols = [c for c in all_cols if c.__dict__[split_by] in attributes_test]  # test cols

        print('\nTraining set:', len(attributes_train), 'unique attributes [', split_by, '],', len(self.train_cols),
              'columns')
        print('Test set:', len(attributes_test), 'unique attributes [', split_by, '],', len(self.test_cols), 'columns')

        labels_train = list(OrderedDict.fromkeys(x.title for x in self.train_cols))

        self.labels = labels_train  # known labels are those from the training set of columns
        if 'unknown' not in self.labels:
            print('Adding \'unknown\' semantic label to the list of labels')
            self.labels = np.append(self.labels, 'unknown')
            print('Updated semantic labels:')
            print(self.labels)
        self.label_lookup = {k: v for v, k in enumerate(self.labels)}
        self.inverted_lookup = {v: k for k, v in self.label_lookup.items()}

        # Replace labels in self.test_cols, that are not in self.labels, with 'unknown':
        for i in range(len(self.test_cols)):
            if self.test_cols[i].title not in self.labels:
                self.test_cols[i].title = 'unknown'

        # Sample from self.train_cols and self.test_cols, mapping labels to label indices:
        (self.X_train['raw'], self.y_train), (_, _), _, _, _ = museum_reader.to_ml(  # sample from self.train_cols
            self.train_cols, self.labels, hp['subsize'], hp['n_samples'], 1.0, add_header=self.add_headers, p_header=self.p_header, verbose=False
        )

        (self.X_test['raw'], self.y_test), (_, _), _, _, _ = museum_reader.to_ml(  # sample from self.test_cols
            self.test_cols, self.labels, hp['subsize'], hp['n_samples'], 1.0, add_header=self.add_headers, p_header=1, verbose=False
        )

        # Further split (X_train, y_train) into (X_train, y_train) and (X_valid, y_valid):
        (self.X_train['raw'], self.y_train), (self.X_valid['raw'], self.y_valid) = train_validation_split(
            self.X_train['raw'], self.y_train, hp['samples_validation_frac'])
        print('len(X_train[\'raw\']):', len(self.X_train['raw']))
        print('len(X_valid[\'raw\']):', len(self.X_valid['raw']))
        print('len(X_test[\'raw\']):', len(self.X_test['raw']))

        print('Semantic labels in training set:  ', np.unique(self.y_train))
        print('Semantic labels in validation set:', np.unique(self.y_valid))
        print('Semantic labels in testing set:   ', np.unique(self.y_test))
        if not set(self.y_train) == set(self.y_valid):
            print(
                "WARNING: validation set does not have same set of semantic labels as training set! This might cause ensemble models to perform poorly.")

        print('\n\nGenerating inputs for classifiers...')
        self._generate_features(self.X_train, '_train')
        self._generate_features(self.X_valid, '_valid')
        self._generate_features(self.X_test, '_test')

        self.y_train = np.array(self.y_train)
        self.y_valid = np.array(self.y_valid)
        self.y_test = np.array(self.y_test)
        self.y_train_binary = to_categorical(self.y_train, len(self.labels))
        self.y_valid_binary = to_categorical(self.y_valid, len(self.labels))
        self.y_test_binary = to_categorical(self.y_test, len(self.labels))

    def train(self, evaluate_after_training=False, verbose=False):
        """Train classifiers specified in self.classifier_types"""
        for t in self.classifier_types:
            print('\n' + '-' * 80)
            print('\nTraining a', t, 'classifier...')

            if t.split('@')[0] == 'cnn':
                self.classifiers[t] = CNN({**hp, **hp_cnn})
                self.classifiers[t].build(n_classes=len(self.labels))
                self.classifiers[t].summary()
                self.classifiers[t].train(self.X_train[t.split('@')[-1]], self.y_train_binary,
                                          self.X_valid[t.split('@')[-1]], self.y_valid_binary)

                # Evaluate after training:
                # TO DO: replace with self.evaluate method?
                if evaluate_after_training:
                    if len(self.X_train[t.split('@')[-1]])>0:
                        print('Evaluating', t, 'on the training set...')
                        performance = self.classifiers[t].evaluate(self.X_train[t.split('@')[-1]], self.y_train_binary)
                        print(' ' * 3 + 'loss:', performance[0])
                        for i, m in enumerate(self.classifiers[t].metrics, start=1):
                            print(' ' * 3 + m, ':', performance[i])

                    if len(self.X_valid[t.split('@')[-1]])>0:
                        print('\nEvaluating', t, 'on the validation set...')
                        performance = self.classifiers[t].evaluate(self.X_valid[t.split('@')[-1]], self.y_valid_binary)
                        print(' ' * 3 + 'loss:', performance[0])
                        for i, m in enumerate(self.classifiers[t].metrics, start=1):
                            print(' ' * 3 + m, ':', performance[i])

                    if len(self.X_test[t.split('@')[-1]])>0:
                        print('\nEvaluating', t, 'on the testing set...')
                        performance = self.classifiers[t].evaluate(self.X_test[t.split('@')[-1]], self.y_test_binary)
                        print(' ' * 3 + 'loss:', performance[0])
                        for i, m in enumerate(self.classifiers[t].metrics, start=1):
                            print(' ' * 3 + m, ':', performance[i])

                # Add 'charseq_embedded' and 'augmented' features:
                if any(f in [t.split('@')[-1] for t in self.classifier_types] for f in ['charseq_embedded']):
                    self.cnn_embedder = CNN_embedder(self.classifiers['cnn@charseq'])
                    self.cnn_embedder.summary()
                    self.X_train['charseq_embedded'] = self.cnn_embedder.predict(self.X_train['charseq'])
                    self.X_valid['charseq_embedded'] = self.cnn_embedder.predict(self.X_valid['charseq'])
                    self.X_test['charseq_embedded'] = self.cnn_embedder.predict(self.X_test['charseq'])

                if any(f in [t.split('@')[-1] for t in self.classifier_types] for f in
                       ['augmented']):
                    try:
                        self.X_train['augmented'] = np.concatenate(
                            (self.X_train['charseq_embedded'], self.X_train['charfreq']), axis=1)
                        self.X_valid['augmented'] = np.concatenate(
                            (self.X_valid['charseq_embedded'], self.X_valid['charfreq']), axis=1)
                        self.X_test['augmented'] = np.concatenate(
                            (self.X_test['charseq_embedded'], self.X_test['charfreq']), axis=1)
                    except:
                        pass   # leave the 'augmented' features at None

            elif t.split('@')[0] == 'rf':
                self.classifiers[t] = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1)
                # train the rf on the training set (validation set will only be used for model ensembling):
                self.classifiers[t].fit(self.X_train[t.split('@')[-1]], self.y_train)

                # Evaluate after training:
                if evaluate_after_training:
                    oob_acc = self.classifiers[t].oob_score_
                    print('OOB accuracy = ', oob_acc)

                    if len(self.X_test[t.split('@')[-1]])>0:
                        y_pred = self.classifiers[t].predict(self.X_test[t.split('@')[-1]])
                        if 'categorical_accuracy' in metrics:
                            test_acc = sklearn.metrics.accuracy_score(self.y_test, y_pred)
                            print('Test accuracy = ', test_acc)
                        if 'fmeasure' in metrics:
                            test_fmeasure = sklearn.metrics.f1_score(self.y_test, y_pred, average=metrics_average)
                            print('Test fmeasure = ', test_fmeasure)
                        # if 'MRR' in metrics:
                        #     y_pred_proba = self.classifiers[t].predict_proba(self.X_test[t.split('@')[-1]])
                        #     test_mrr = sklearn.metrics.label_ranking_average_precision_score(self.y_test_binary, y_pred_proba)
                        #     print('Test MRR = ',test_mrr)

            elif t.split('@')[0] == 'mlp':
                self.classifiers[t] = MLP({**hp, **hp_mlp})
                self.classifiers[t].build(input_dim=self.X_train[t.split('@')[-1]].shape[1], n_classes=len(self.labels))
                self.classifiers[t].summary()
                self.classifiers[t].train(self.X_train[t.split('@')[-1]], self.y_train_binary,
                                          self.X_valid[t.split('@')[-1]], self.y_valid_binary)

                # Evaluate after training:
                if evaluate_after_training:
                    if len(self.X_train[t.split('@')[-1]])>0:
                        print('Evaluating', t, 'on the training set...')
                        performance = self.classifiers[t].evaluate(self.X_train[t.split('@')[-1]], self.y_train_binary)
                        print(' ' * 3 + 'loss:', performance[0])
                        for i, m in enumerate(labeler.classifiers[t].metrics, start=1):
                            print(' ' * 3 + m, ':', performance[i])

                    if len(self.X_valid[t.split('@')[-1]])>0:
                        print('\nEvaluating', t, 'on the validation set...')
                        performance = self.classifiers[t].evaluate(self.X_valid[t.split('@')[-1]], self.y_valid_binary)
                        print(' ' * 3 + 'loss:', performance[0])
                        for i, m in enumerate(labeler.classifiers[t].metrics, start=1):
                            print(' ' * 3 + m, ':', performance[i])

                    if len(self.X_test[t.split('@')[-1]])>0:
                        print('\nEvaluating', t, 'on the testing set...')
                        performance = self.classifiers[t].evaluate(self.X_test[t.split('@')[-1]], self.y_test_binary)
                        print(' ' * 3 + 'loss:', performance[0])
                        for i, m in enumerate(labeler.classifiers[t].metrics, start=1):
                            print(' ' * 3 + m, ':', performance[i])

    def predict_proba(self, cols, verbose=False):
        """Predict semantic label probabilities for columns in cols, using self.classifiers"""
        predictions = []
        for i, col in enumerate(cols):
            predictions.append({t: None for t in self.classifier_types})
            if verbose: print("Predicting label probabilities for column", i + 1, "out of", len(cols))
            X_query = {}
            (X_query['raw'], _), (_, _), _, _, _ = museum_reader.to_ml(  # sample from col
                [col], self.labels, hp['subsize'], hp['n_samples'], 1.0, add_header=self.add_headers, p_header=1, verbose=False
            )

            # Prepare input for classifiers:
            self._generate_features(X_query, '_query')  # after this, X_query should have feature matrix of samples
            if verbose: print('Generated features of X_query:', X_query.keys())
            # Loop over self.classifiers:
            # For each classifier, predict hard label of col by argmax of mean soft label predictions of all rows in X_query
            for t, classifier in self.classifiers.items():
                if verbose: print(" " * 3 + "using", t, "classifier...", end=" ")

                y_soft_pred = self._predict_soft(classifier, X_query[t.split('@')[-1]])

                predictions[i][t] = y_soft_pred
                if verbose: print("done")

        return predictions

    def predict(self, cols, verbose=False):
        """Predict semantic labels of columns in cols, using self.classifiers"""
        predictions = []
        for i, col in enumerate(cols):
            predictions.append({t: None for t in self.classifier_types})
            if verbose: print("Predicting label for column", i + 1, "out of", len(cols))
            X_query = {}
            (X_query['raw'], _), (_, _), _, _, _ = museum_reader.to_ml(  # sample from col
                [col], self.labels, hp['subsize'], hp['n_samples'], 1.0, add_header=self.add_headers, p_header=1, verbose=False
            )

            # Prepare input for classifiers:
            self._generate_features(X_query, '_query')  # after this, X_query should have feature matrix of samples
            if verbose: print('Generated features of X_query:', X_query.keys())
            # Loop over self.classifiers:
            # For each classifier, predict hard label of col by argmax of mean soft label predictions of all rows in X_query
            for t, classifier in self.classifiers.items():
                if verbose: print(" " * 3 + "using", t, "classifier...", end=" ")

                y_soft_pred = self._predict_soft(classifier, X_query[t.split('@')[-1]])
                y_pred, label_pred = self._predict_hard(y_soft_pred)

                predictions[i][t] = label_pred
                if verbose: print("done. Predicted label:", label_pred)

        return predictions

    def evaluate(self, cols, verbose=False):
        """Evaluate performance metrics (e.g., mean accuracy) of classifiers in self.classifiers predicting labels for cols"""
        performance = {}
        for m in metrics:
            performance[m] = OrderedDict()

        y_true = np.array([c.title for c in cols])
        y_pred = self.predict(cols,
                              verbose)  # this is a list of dictionaries with key=classifier_type, value=predicted label
        y_true_proba = to_categorical(np.array([self.label_lookup[y] for y in y_true]), nb_classes=len(self.labels))
        y_pred_proba = self.predict_proba(cols,
                                          verbose)  # this is a list of dictionaries with key=classifier_type, value=predicted probability scores of all classes

        for t in self.classifier_types:
            y_pred_t = np.array([y[t] for y in y_pred])  # extract predictions by classifier t for all cols
            if 'categorical_accuracy' in metrics:
                performance['categorical_accuracy'][t] = sklearn.metrics.accuracy_score(y_true,
                                                                                        y_pred_t)  # np.mean(y_pred_t == y_true)
            if 'fmeasure' in metrics:
                performance['fmeasure'][t] = sklearn.metrics.f1_score(y_true, y_pred_t, average=metrics_average)

            if 'MRR' in metrics:
                y_pred_proba_t = np.array(
                    [y[t] for y in y_pred_proba])  # extract predictions by classifier t for all cols
                performance['MRR'][t] = sklearn.metrics.label_ranking_average_precision_score(y_true_proba,
                                                                                              y_pred_proba_t)

        return performance


class Ensemble_Average(object):
    """Ensemble models (classifiers in labeler and, optionally, Paul's model in paul_labeler) via unweighted averaging of their class probability predictions"""

    def __init__(self, labeler, paul_labeler=None):
        self.labeler = labeler
        self.paul_labeler = paul_labeler
        if paul_labeler is not None: self.paul_predictions_file = paul_labeler.predictions_file

    def predict_proba(self, query_cols):
        """Predict average class probabilities, by averaging class probabilities of all models in self.labeler and self.paul_labeler"""
        predictions_proba = self.labeler.predict_proba(
            query_cols)  # this is a list of dicts, each dict has key=model, value=np.array(class probabilities)
        try:
            predictions_proba_paul, _ = self.paul_labeler.predict(query_cols,
                                                                  self.paul_predictions_file)  # this is a list of dicts, each dict has key=semantic class, value=class probability
            # Convert predictions_proba_paul to the same format as predictions_proba, and add to predictions_proba:
            for i, paul_pred in enumerate(predictions_proba_paul):
                y_pred = np.array([paul_pred[l] for l in [self.labeler.inverted_lookup[i] for i in range(len(
                    self.labeler.labels))]])  # Paul's model's predicted class probabilities for column query_cols[i]
                predictions_proba[i][
                    'paul'] = y_pred  # add Paul's model's predicted class probabilities to predictions_proba
        except:
            pass  # if paul_labeler is not passed as an argument, do nothing here, and just ensemble predictions_proba by 'my' classifiers only

        for p in predictions_proba:  # loop over predictions_proba for query_cols
            p['ensemble_avg'] = np.mean(np.dstack([p.popitem()[1] for _ in range(len(p.items()))]),
                                        axis=2)  # stack all probability vectors along 3rd dim, and then take the average across the stack

        return predictions_proba

    def predict(self, query_cols):
        """Predict class label using the result of predict_proba"""
        predictions_proba = self.predict_proba(query_cols)
        y_pred = []
        for p in predictions_proba:
            y_pred.append(self.labeler.inverted_lookup[np.argmax(p['ensemble_avg'])])

        return y_pred

    def evaluate(self, cols):
        """Evaluate performance of the ensemble, using cols as the test set"""
        performance = {}
        for m in metrics:
            performance[m] = OrderedDict()

        y_true = np.array([c.title for c in cols])
        y_pred = np.array(self.predict(cols))
        y_true_proba = to_categorical(np.array([self.labeler.label_lookup[y] for y in y_true]),
                                      nb_classes=len(self.labeler.labels))
        y_pred_proba = np.array([p['ensemble_avg'].flatten() for p in self.predict_proba(cols)])

        if 'categorical_accuracy' in metrics:
            performance['categorical_accuracy'] = sklearn.metrics.accuracy_score(y_true,
                                                                                 y_pred)  # np.mean(y_pred == y_true)
        if 'fmeasure' in metrics:
            performance['fmeasure'] = sklearn.metrics.f1_score(y_true, y_pred, average=metrics_average)
        if 'MRR' in metrics:
            performance['MRR'] = sklearn.metrics.label_ranking_average_precision_score(y_true_proba, y_pred_proba)

        return performance, y_pred, y_true


if __name__ == "__main__":

    # ## Read the data columns
    # reader for the museum dataset
    data_dir = 'data/museum/'
    museum_reader = Reader()
    files, all_cols = museum_reader.read_dir(data_dir)
    print("Found", len(files), "files (sources) with a total of", len(all_cols), "columns in", data_dir)

    for c in all_cols:
        c.id = c.title + '@' + c.filename

    # TO DO: Split files (data sources) into train and test sets of files (sources), rather than splitting columns (after putting them all together) into train and test sets of columns:
    all_cols[0].__dict__.keys()

    labeler = NN_Column_Labeler(
        ['cnn@charseq', 'rf@charseq_embedded', 'mlp@charfreq', 'rf@charfreq', 'mlp@augmented', 'rf@augmented'],
        all_cols, split_by=hp['split_by'], test_frac=hp['cols_test_frac'])

    labels_train = list(OrderedDict.fromkeys(x.title for x in labeler.train_cols))
    if 'unknown' in labels_train:
        print('WARNING: \'unknown\' is in labels_train, which should not happen for the museum dataset!')

    # # Relative fractions of semantic classes in the training set:
    # for y in np.unique(labeler.y_train):
    #     y_frac = np.sum(labeler.y_train == y)/len(labeler.y_train)
    #     print(y,':',round(y_frac,3),labeler.inverted_lookup[y])

    labeler.train()

    query_cols = copy.deepcopy(labeler.test_cols)
    print('Evaluating on', len(query_cols), 'test columns...')
    # label_pred = labeler.predict(query_cols, verbose=False)
    performance = labeler.evaluate(query_cols)

    ensemble_labeler = Ensemble_Average(labeler)  # ensemble classifiers in labeler by averaging their proba predictions
    performance_ensemble_labeler, _, _ = ensemble_labeler.evaluate(query_cols)

    for k in performance_ensemble_labeler.keys():
        performance[k]['ensemble_avg_my'] = performance_ensemble_labeler[k]

    performances = {}  # use this to collect results of runs
    for m in metrics:
        performances[m] = []  # use this to collect results of runs

    for m in metrics:
        performances[m].append(performance[m])

    # results_dir = './results/'
    # fname_progress = results_dir+'labeler_performances, data=museum, data_folds='+', model_folds='+' [IN PROGRESS].xlsx'
    # writer = ExcelWriter(fname_progress)
    performances_df = {}
    for m in metrics:
        performances_df[m] = pd.DataFrame.from_dict(performances[m])
        # Save the progress:
    #     performances_df[m].to_excel(excel_writer=writer,sheet_name=m, index=False)

    # writer.save()

    print(performances_df['categorical_accuracy'])

    print(performances_df['fmeasure'])

    print(performances_df['MRR'])
