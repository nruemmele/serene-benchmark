from benchmark import Experiment, NNetModel
import os
import logging


def test_cnn():
    # ******* setting up NNetModel
    nnet_model = NNetModel(['cnn@charseq'], 'cnn@charseq model', add_headers=True, p_header=0,
                           debug_csv=os.path.join("data", "debug_nnet_cnn.csv"))
    # nnet_model = NNetModel(['rf@charfreq'], 'rf@charfreq model', add_headers=False)

    # models for experiments
    models = [nnet_model]
    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('data', "performance_cnn.csv"),
                                debug_csv=os.path.join("data", "debug_cnn.csv"))
    loo_experiment.run()



if __name__ == "__main__":

    # setting up the logging
    log_file = 'benchmark_cnn.log'
    # log_format = "%(asctime)s [%(name)-12.12s] [%(levelname)-10.10s]  %(message)s"
    # log_format = "%(asctime)s [%(filename)s:%(lineno)s  %(funcName)20s() ] %(message)s"
    log_format = "%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s"
    logging.basicConfig(filename=os.path.join('data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format=log_format, datefmt='%m/%d/%Y %I:%M:%S %p')


    test_cnn()

