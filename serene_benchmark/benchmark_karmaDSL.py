"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

"""

from serene_benchmark import Experiment, KarmaDSLModel
import os
from karmaDSL import KarmaSession
import logging
from logging.handlers import RotatingFileHandler


def test_karma(ignore_unknown=True, experiment_type="leave_one_out", domains=None):
    # ******** setting up KarmaDSL model
    dsl = KarmaSession(host="localhost", port=8000)
    dsl_model1 = KarmaDSLModel(dsl, "default KarmaDSL model",
                               ignore_unknown=ignore_unknown, prediction_type="column",
                               debug_csv=os.path.join("results",
                                                      "debug_dsl_column_{}_ignore{}.csv".format(
                                                          experiment_type, ignore_unknown)
                                                      ))
    dsl_model2 = KarmaDSLModel(dsl, "KarmaDSL",
                               ignore_unknown=ignore_unknown, prediction_type="folder",
                               debug_csv=os.path.join("results",
                                                      "debug_dsl_folder_{}_ignore{}.csv".format(
                                                          experiment_type, ignore_unknown)
                                                      ))
    dsl_model_plus = KarmaDSLModel(dsl, "KarmaDSL+",
                               ignore_unknown=ignore_unknown, prediction_type="enhanced",
                               debug_csv=os.path.join("results",
                                                      "debug_dsl_column_{}_ignore{}.csv".format(
                                                          experiment_type, ignore_unknown)
                                                      ))

    # models for experiments
    models = [dsl_model1, dsl_model2]
    models = [dsl_model1]

    experiment = Experiment(models,
                            experiment_type=experiment_type,
                            description=experiment_type,
                            result_csv=os.path.join('results', "performance_dsl_{}_{}.csv".format(experiment_type, "all")),
                            debug_csv=os.path.join("results", "debug_dsl_{}_{}.csv".format(experiment_type, "all")),
                            holdout=0.2, num=1
                            )

    if domains:
        experiment.change_domains(domains)
    experiment.run()

if __name__ == "__main__":

    # setting up the logging
    log_file = os.path.join("results", 'benchmark_dsl.log')
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

    # experiments = ["leave_one_out", "repeated_holdout"]
    #
    # for exp in experiments:
    #     print("Performing experiment:", exp)
    #     test_karma(experiment_type=exp, domains=["weather"])

    experiments = [ "leave_one_out"]
    # experiments = ["repeated_holdout"]
    # ignore = [False, True]
    ignore = [True]

    for ig in ignore:
        print("Setting ignore_unknown: ", ig)
        for exp in experiments:
            print("Performing experiment:", exp)
            test_karma(ignore_unknown=ig, experiment_type=exp)

    ignore = [False]
    for ig in ignore:
        print("Setting ignore_unknown: ", ig)
        for exp in experiments:
            print("Performing experiment:", exp)
            test_karma(ignore_unknown=ig, experiment_type=exp, domains=["soccer", "museum"])

