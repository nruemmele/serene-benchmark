
import logging
import os
import time
import numpy as np
import pandas as pd

from neural_nets import hp, hp_cnn
from benchmark.semantic_typer import NNetModel, domains, benchmark


# Limit the allocated GPU memory to a fraction of total GPU memory, as per https://groups.google.com/forum/#!topic/keras-users/MFUEY9P1sc8:
# This is for sharing GPU with other sessions/users
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
def get_session(gpu_fraction=0.5):
    '''Allocate a specified fraction of GPU memory for keras tf session'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def experiment(domain, classifier_type, model_description, add_headers, p_step, n_runs, results_dir):

    # #******* setting up DINTModel
    # dm = SchemaMatcher(host="localhost", port=8080)
    # # dictionary with features
    # feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals"],
    #             "activeFeatureGroups": ["stats-of-text-length", "prop-instances-per-class-in-knearestneighbours"],
    #             "featureExtractorParams": [
    #                 {"name": "prop-instances-per-class-in-knearestneighbours", "num-neighbours": 5}]
    #             }
    # # resampling strategy
    # resampling_strategy = "ResampleToMean"
    # dint_model = DINTModel(dm, feature_config, resampling_strategy, "DINTModel with ResampleToMean")
    #
    # print("Define training data DINT %r" % dint_model.define_training_data(train_sources))
    # print("Train dint %r" % dint_model.train())
    # predicted_df = dint_model.predict(test_source[0])
    # print(predicted_df)
    # print(dint_model.evaluate(predicted_df))
    #
    # # ******** setting up KarmaDSL model
    # dsl = KarmaSession(host="localhost", port=8000)
    # dsl_model = KarmaDSLModel(dsl, "default KarmaDSL model")
    #
    # print("Define training data KarmaDSL %r" % dsl_model.define_training_data(train_sources))
    # print("Train dsl %r" % dsl_model.train())
    # predicted_df = dsl_model.predict(test_source[0])
    # print(predicted_df)
    # print(dsl_model.evaluate(predicted_df))

    if add_headers:
        p_header_list = np.arange(0., 1. + p_step, p_step)  # range from 0. to 1. with p_step
    else:
        p_header_list = [0.]

    results_file = 'adding_headers_to_samples ' + '(' + domain + ', ' + model_description + ', ' + str(n_runs) + ' runs per p_header value)'
    if not add_headers:
        results_file = 'not '+results_file
    fname_progress = results_dir + results_file + ' [IN PROGRESS].xlsx'

    logging.info("Experiment on probabilistic inclusion of column headers to column samples")
    logging.info('Parameters of the experiment:')
    logging.info('  domain: {}, classifier_type: {}'.format(domain, classifier_type))
    logging.info('  model description: {}'.format(model_description))
    logging.info('  add_headers={}, p_step={}'.format(add_headers, p_step))
    logging.info('RESULTS will be saved to file \'{}\''.format(results_file))

    results = pd.DataFrame(columns=['runs','add_header','p_header','accuracy_mean','accuracy_std','fmeasure_mean','fmeasure_std','MRR_mean','MRR_std'])
    for p_header in p_header_list:
        for run in range(n_runs):
            logging.info("p_header={}, run {} of {}...".format(p_header,run+1,n_runs))
            if classifier_type=='cnn@charseq':
                K.clear_session()  # Destroys the current TF graph and creates a new one. Useful to avoid clutter from old models / layers.
            nnet_model = NNetModel([classifier_type], model_description, add_headers=add_headers, p_header=p_header)
            nnet_model.define_training_data(train_sources)
            # Train the nnet_model:
            nnet_model.train()

            predictions = nnet_model.predict(test_source)

            if run==0:
                performance = nnet_model.evaluate(predictions)
            else:
                performance = performance.append(nnet_model.evaluate(predictions))

        performance_mean = (performance.mean(axis=0, numeric_only=True))
        performance_std = (performance.std(axis=0, numeric_only=True))
        print("\nPERFORMANCE:")
        print(performance)
        print("\nMEAN:")
        print(performance_mean)
        print("\nSTD:")
        print(performance_std)

        results_row = [{'runs':n_runs,'add_header':add_headers,'p_header':p_header,'accuracy_mean':performance_mean['categorical_accuracy'],'accuracy_std':performance_std['categorical_accuracy'],
                        'fmeasure_mean':performance_mean['fmeasure'],'fmeasure_std':performance_std['fmeasure'],'MRR_mean':performance_mean['MRR'],'MRR_std':performance_std['MRR']}]
        results = results.append(results_row, ignore_index=True)

        writer = pd.ExcelWriter(fname_progress)
        results.to_excel(excel_writer=writer, index=False)   # save the progress so far
        writer.save()

    fname_final = results_dir + results_file + ' ['+time.strftime("%d-%m-%Y")+'].xlsx'
    os.rename(fname_progress, fname_final)
    logging.info("Results are saved to file {}".format(fname_final))

    print("\n\nEXPERIMENT RESULTS:")
    print(results)


if __name__ == "__main__":
    print('EXPERIMENT')
    # setting up the logging
    log_file = 'benchmark.log'
    logging.basicConfig(filename=os.path.join('data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    results_dir = '/home/yuriy/Projects/Data_integration/code/serene-benchmark/benchmark/experiments/'

    # ******* setting up a bunch of experiments:
    domain = "soccer"
    assert domain in domains
    # train/test
    train_sources = benchmark[domain][:-1]
    test_source = [benchmark[domain][-1]]
    print("Domain:", domain)
    print("# sources in train: %d" % len(train_sources))
    print("# sources in test: %d" % len(test_source))

    trials = {}
    n_experiments = 5  # each experiment should check some hypothesis
    n_trials = n_experiments*2   # *2 since we have pairs of trials with add_headers=[False, True] for each experiment
    trials['classifier_type'] = ['cnn@charseq']*n_trials
    trials['add_headers'] = [False,True]*n_experiments
    trials['n_conv_layers'] = [2]*2 + [2]*2 + [3]*2 + [2]*2 + [2]*2
    trials['nb_filter'] = [100]*2 + [100]*2 + [100]*2 + [200]*2 + [200]*2
    trials['maxlen'] = [250]*2 + [500]*2 + [250]*2 + [200]*2 + [250]*2

    p_step = 0.1
    n_runs = 100   # models per value of p, to be averaged

    for t in range(n_trials):
        # Specify the parameters of the run:
        classifier_type = trials['classifier_type'][t]
        add_headers = trials['add_headers'][t]
        hp_cnn['n_conv_layers'] = trials['n_conv_layers'][t]
        hp_cnn['nb_filter'] = trials['nb_filter'][t]
        hp['maxlen'] = trials['maxlen'][t]

        if classifier_type == 'cnn@charseq':
            KTF.set_session(get_session())
            model_description = classifier_type + ' model with ' + \
                str(hp_cnn['n_conv_layers']) + ' conv layers of ' + \
                str(hp_cnn['nb_filter']) + ' filters, ' + 'charseq_length=' + str(hp['maxlen'])
        else:
            model_description = classifier_type + ' model'

        experiment(domain, classifier_type, model_description, add_headers, p_step, n_runs, results_dir)