"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

"""

from serene_benchmark import Experiment, KarmaDSLModel
import os
from karmaDSL import KarmaSession
import logging

def test_karma():
    # ******** setting up KarmaDSL model
    dsl = KarmaSession(host="localhost", port=8000)
    dsl_model = KarmaDSLModel(dsl, "default KarmaDSL model")

    # models for experiments
    models = [dsl_model]

    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_karma.csv"),
                                debug_csv=os.path.join("results", "debug_karma.csv"))

    loo_experiment.run()

if __name__ == "__main__":

    # setting up the logging
    log_file = 'benchmark_karma.log'
    logging.basicConfig(filename=os.path.join('results', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    test_karma()