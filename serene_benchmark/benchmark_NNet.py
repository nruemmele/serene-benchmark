"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

"""

from serene_benchmark import Experiment, NNetModel
import os
import logging


def test_cnn():
    # ******* setting up NNetModel
    cnn_model = NNetModel(['cnn@charseq'], 'cnn@charseq model: no headers',
                           add_headers=False, p_header=0,
                           debug_csv=os.path.join("results", "debug_nnet_cnn.csv"))
    cnn_model_head = NNetModel(['cnn@charseq'], 'cnn@charseq model: headers, p=0.4',
                           add_headers=True, p_header=0.4,
                           debug_csv=os.path.join("results", "debug_nnet_cnn_head.csv"))

    rf_model = NNetModel(['rf@charfreq'], 'rf@charfreq model: no headers', add_headers=False,  p_header=0,
                           debug_csv=os.path.join("results", "debug_nnet_rf.csv"))

    # models for experiments
    models = [rf_model, cnn_model]
    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_cnn.csv"),
                                debug_csv=os.path.join("results", "debug_cnn.csv"))
    loo_experiment.run()



if __name__ == "__main__":

    # setting up the logging
    log_file = 'benchmark_nnet.log'
    # log_format = "%(asctime)s [%(name)-12.12s] [%(levelname)-10.10s]  %(message)s"
    # log_format = "%(asctime)s [%(filename)s:%(lineno)s  %(funcName)20s() ] %(message)s"
    log_format = "%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s"
    logging.basicConfig(filename=os.path.join('results', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format=log_format, datefmt='%m/%d/%Y %I:%M:%S %p')


    test_cnn()

