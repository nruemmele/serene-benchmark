"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

"""

from serene_benchmark import Experiment, NNetModel
import os
import logging
from logging.handlers import RotatingFileHandler


def test_cnn(ignore_unknown=True, experiment_type="leave_one_out", domains=None):
    # ******* setting up NNetModel
    cnn_model = NNetModel(['cnn@charseq'], 'cnn@charseq model: no headers',
                           add_headers=False, p_header=0,
                           debug_csv=os.path.join("results", "debug_nnet_cnn_ignore{}_{}.csv".format(
                                ignore_unknown, experiment_type)),
                          ignore_unknown=ignore_unknown)
    mlp_model = NNetModel(['mlp@charfreq'], 'mlp@charfreq model: no headers',
                          add_headers=False, p_header=0,
                          debug_csv=os.path.join("results", "debug_nnet_mlp_ignore{}_{}.csv".format(
                                ignore_unknown, experiment_type)),
                          ignore_unknown=ignore_unknown)
    cnn_model_head = NNetModel(['cnn@charseq'], 'cnn@charseq model: headers, p=0.4',
                               add_headers=True, p_header=0.4,
                               debug_csv=os.path.join("results", "debug_nnet_cnn_head_ignore{}_{}.csv".format(
                                ignore_unknown, experiment_type)),
                               ignore_unknown=ignore_unknown)

    rf_model = NNetModel(['rf@charfreq'], 'rf@charfreq model: no headers', add_headers=False,  p_header=0,
                         debug_csv=os.path.join("results", "debug_nnet_rf_ignore{}_{}.csv".format(
                                ignore_unknown, experiment_type)),
                         ignore_unknown=ignore_unknown)

    # models for experiments
    models = [rf_model, cnn_model, mlp_model]
    experiment = Experiment(models,
                            experiment_type=experiment_type,
                            description=experiment_type+"_ignore"+str(ignore_unknown),
                            result_csv=os.path.join('results', "performance_nnet_ignore{}_{}.csv".format(
                                ignore_unknown, experiment_type)),
                            debug_csv=os.path.join("results", "debug_nnet_ignore{}_{}.csv".format(
                                ignore_unknown, experiment_type)),
                            holdout=0.2, num=1)

    if domains:
        experiment.change_domains(domains)
    experiment.run()

if __name__ == "__main__":

    # setting up the logging
    log_file = os.path.join("results",'benchmark_nnet.log')
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s: %(message)s',
                                      '%Y-%m-%d %H:%M:%S')
    my_handler = RotatingFileHandler(log_file, mode='a', maxBytes=5 * 1024 * 1024,
                                     backupCount=5, encoding=None, delay=0)
    my_handler.setFormatter(log_formatter)
    my_handler.setLevel(logging.DEBUG)
    # logging.basicConfig(filename=log_file,
    #                     level=logging.DEBUG, filemode='w+',
    #                     format='%(asctime)s %(levelname)s %(module)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(my_handler)

    experiments = ["leave_one_out", "repeated_holdout"]
    experiments = ["repeated_holdout"]

    ignore = [True]
    for ig in ignore:
        print("Setting ignore_unknown: ", ig)
        for exp in experiments:
            print("Performing experiment:", exp)
            test_cnn(ignore_unknown=ig, experiment_type=exp, domains=["weather"])

    # ignore = [False]
    # for ig in ignore:
    #     print("Setting ignore_unknown: ", ig)
    #     for exp in experiments:
    #         print("Performing experiment:", exp)
    #         test_cnn(ignore_unknown=ig, experiment_type=exp, domains=["soccer", "museum"])



