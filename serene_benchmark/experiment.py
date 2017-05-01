"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Module for experiment design
"""
import logging
import os

import pandas as pd
import random
from collections import defaultdict
from serene.api.exceptions import InternalError


class Experiment(object):
    """

    """
    # this is the path to benchmark resources
    if "SERENEPATH" in os.environ:
        data_dir = os.path.join(os.environ["SERENEPATH"], "sources")
        label_dir = os.path.join(os.environ["SERENEPATH"], "labels")
    else:
        data_dir = os.path.join("data", "sources")
        label_dir = os.path.join("data", "labels")

    domains = ["soccer", "dbpedia", "museum", "weather", "weapons"]
    benchmark = {
        "soccer": ['bundesliga-2015-2016-rosters.csv', 'world_cup_2010_squads.csv',
                   'fifa-soccer-12-ultimate-team-data-player-database.csv', 'world_cup_2014_squads.csv',
                   'all_world_cup_players.csv', 'mls_players_2015.csv', 'world_cup_squads.csv',
                   'players.csv', '2014 WC french.csv', 'uefa_players.csv', 'world_cup_player_ages.csv',
                   'WM 2014 Alle Spieler - players fifa.csv'],
        "dbpedia": ['s5.txt', 's2.txt', 's6.txt',
                    's3.txt', 's4.txt', 's9.txt',
                    's8.txt', 's7.txt', 's10.txt', 's1.txt'],
        "museum": ['s02-dma.csv', 's26-s-san-francisco-moma.json', 's27-s-the-huntington.json',
                   's16-s-hammer.xml', 's20-s-lacma.xml', 's15-s-detroit-institute-of-art.json',
                   's28-wildlife-art.csv', 's04-ima-artworks.xml', 's25-s-oakland-museum-paintings.json',
                   's29-gilcrease.csv', 's05-met.json', 's13-s-art-institute-of-chicago.xml',
                   's14-s-california-african-american.json', 's07-s-13.json', 's21-s-met.json',
                   's12-s-19-artworks.json', 's08-s-17-edited.xml', 's19-s-indianapolis-artworks.xml',
                   's11-s-19-artists.json', 's22-s-moca.xml', 's17-s-houston-museum-of-fine-arts.json',
                   's18-s-indianapolis-artists.xml', 's23-s-national-portrait-gallery.json',
                   's03-ima-artists.xml', 's24-s-norton-simon.json', 's06-npg.json',
                   's09-s-18-artists.json', 's01-cb.csv'],
        "weather": ['w1.txt', 'w3.txt', 'w2.txt', 'w4.txt'],
        "weapons": ["www.theoutdoorstrader.com.csv", "www.tennesseegunexchange.com.csv",
                    "www.shooterswap.com.csv", "www.nextechclassifieds.com.csv", "www.msguntrader.com.csv",
                    "www.montanagunclassifieds.com.csv", "www.kyclassifieds.com.csv",
                    "www.hawaiiguntrader.com.csv", "www.gunsinternational.com.csv",
                    "www.floridaguntrader.com.csv", "www.floridagunclassifieds.com.csv",
                    "www.elpasoguntrader.com.csv", "www.dallasguns.com.csv",
                    "www.armslist.com.csv", "www.alaskaslist.com.csv"
                    ]
    }

    def __init__(self, models, experiment_type, description, result_csv, debug_csv,
                 holdout=None, num=None):
        """
        Initialize experiment.
        To run the experiment, please call "run" explicitly.
        :param models:
        :param experiment_type: "leave_one_out", "repeated_holdout"
        :param description:
        :param result_csv:
        :param debug_csv:
        :param holdout: (0,1) range float number
        :param num: integer > 0
        """
        logging.info("Initializing experiment...")
        self.models = models
        self.experiment_type = experiment_type
        self.description = description
        self.debug_csv = debug_csv
        self.performance_csv = result_csv
        self.train_sources = None
        self.test_sources = None
        if holdout and 0 < holdout < 1:
            self.holdout = holdout
        else:
            self.holdout = 0.5
        if num and num > 0:
            self.num = num
        else:
            self.num = 10

    def add_domain(self, domain_name, sources):
        """
        Add a domain to the benchmark.
        Csv sources and their labels need to be added.
        All source files need to end with .csv.
        All label files need to end with .columnmap.txt.
        Label files have names as source_file_name + .columnmap.txt.
        Source files have names as source_file_name + .csv.
        :param domain_name: name to be assigned to the new domain
        :param sources: list of source_file_name in the domain
        :return:
        """
        logging.info("Adding new domain {}".format(domain_name))
        if not(isinstance(sources, list)):
            logging.error("sources must be a list in add_domain.")
            raise InternalError("add_domain in experiment", "sources must be a list")

        if domain_name in self.domains:
            logging.error("Can't add a domain since this name already exists in the experiment. Change domain_name")
            raise InternalError("add_domain in experiment", "Can't add a domain since this name "
                                                            "already exists in the experiment. Change domain_name")
        avail_sources = os.listdir(self.data_dir)
        avail_labels = os.listdir(self.label_dir)

        new_sources = [s for s in sources
                       if s + ".csv" in avail_sources and s + ".columnmap.txt" in avail_labels]
        if len(new_sources) < len(sources):
            logging.warning("Not all source/label files have been copied to the data_dir/label_dir.")
        logging.info("{} sources will be added to the domain {}".format(len(new_sources), domain_name))
        print("{} sources will be added to the domain {}".format(len(new_sources), domain_name))
        self.benchmark[domain_name] = new_sources
        self.domains.append(domain_name)

    def change_domains(self, domains):
        """
        Change list of domains for evaluation in this experiment.
        :param domains: list of domain names
        :return:
        """
        logging.info("Changing experiment domains. Current domains {}".format(self.domains))
        if not(isinstance(domains, list)):
            logging.error("Domains can't be changed since provided parameter is not a list")
            raise InternalError("change_domains in experiment", "parameter domains must be a list")
        new_domains = [domain for domain in domains if domain in self.domains]
        self.domains = new_domains
        logging.info("New domains in the experiment {}".format(self.domains))

    def _evaluate_model(self, model):
        """
        Evaluate one model by training first on the specified train_sources and testing on the specified test_sources.
        :param model: instance of the class SemanticTyper
        :return:
        """
        if self.train_sources is None:
            logging.error("Train sources are not specified.")
            return None
        if self.test_sources is None:
            logging.error("Test sources are not specified.")
            return None

        frames = []
        try:
            model.reset()
            model.define_training_data(self.train_sources)
            run_time = model.train()
            for test_source in self.test_sources:
                predicted_df1 = model.predict(test_source)
                logging.info("Experiment evaluate: Prediction done.")
                predicted_df1["experiment"] = self.experiment_type
                predicted_df1["experiment_description"] = self.description
                predicted_df1["train_run_time"] = run_time
                # change labels in the test resource to unknown if the correct label is not among those in the training set
                # this is done in SemanticTyper.evaluate() method
                frames.append(predicted_df1)
            predicted_df = pd.concat(frames, ignore_index=True)
            return predicted_df
        except Exception as e:
            logging.warning("Model evaluation failed: {}".format(e))
            return None

    def _leave_one_out(self):
        """
        This experiment does serene_benchmark evaluation for each domain:
            - one dataset is left out for testing
            - models are trained on the remaining datasets in the domain
        :return:
        """
        logging.info("Leave one out experiment")
        performance_frames = []
        for domain in self.domains:
            print("Working on domain: {}".format(domain))
            logging.info("Working on domain: {}".format(domain))
            for model in self.models:
                logging.info("--> evaluating model: {}".format(model))
                frames = []
                for idx, source in enumerate(self.benchmark[domain]):
                    print("----> test source: {}, {}".format(idx, source))

                    logging.info("----> test source: {}".format(source))
                    self.train_sources = self.benchmark[domain][:idx] + self.benchmark[domain][idx+1:]
                    self.test_sources = [source]

                    predicted_df = self._evaluate_model(model)
                    if predicted_df is not None:
                        frames.append(predicted_df)

                if len(frames) > 0:
                    predicted_df = pd.concat(frames)
                    if self.debug_csv:
                        if model.debug_csv:
                            debug_csv = model.debug_csv + "_" + domain + ".csv"
                        else:
                            debug_csv = self.debug_csv + "_" + domain + ".csv"
                        predicted_df.to_csv(debug_csv, index=False, header=True, mode="w+")

                    performance = model.evaluate(predicted_df)
                    performance["train_run_time"] = predicted_df["train_run_time"].mean()
                    performance["experiment"] = self.experiment_type
                    performance["experiment_description"] = self.description
                    performance["predict_run_time"] = predicted_df["running_time"].mean()
                    performance["domain"] = domain
                    if len(frames) == len(self.benchmark[domain]):
                        performance["status"] = "success"
                    else:
                        performance["status"] = "completed_" + str(len(frames))
                    performance["model"] = model.model_type
                    performance["model_description"] = model.description
                else:
                    res = {"model": model.model_type,
                           "model_description": model.description,
                           "experiment": self.experiment_type,
                           "experiment_description": self.description,
                           "status": "failure",
                           "domain": domain}
                    performance = pd.DataFrame(res, index=[0])
                performance_frames.append(performance)
                # print("performance: ", performance)

        performance = pd.concat(performance_frames, ignore_index=True)
        if self.performance_csv:
            if os.path.exists(self.performance_csv):
                performance.to_csv(self.performance_csv, index=False, header=False, mode="a")
            else:
                performance.to_csv(self.performance_csv, index=False, header=True, mode="w+")

        return True

    def _contruct_holdout_samples(self, domain, holdout=0.5):
        """
        Helper function to construct train and test samples based on holdout ratio.
        :param domain: domain
        :param holdout: holdout ratio
        :return: train and test samples
        """
        sources = self.benchmark[domain]
        # we make sure that train_size will leave something for the test sample as well
        train_size = min(max(round(holdout * len(sources)), 1), len(sources)-1)
        train = random.sample(sources, train_size)

        return train, list(set(sources) - set(train))


    def _repeated_holdout(self, holdout=0.5, num=10):
        """
        This experiment does serene_benchmark evaluation for each domain using repeated holdout validation:
            - domain data sources are split into train/test according to holdout
            - we repeat num times random splitting, then training and then prediction
        :return:
        """
        logging.info("Repeated holdout experiment: holdout={}, num={}".format(holdout, num))
        performance_frames = []
        # we just set key on models
        model_lookup = {i: model for i, model in enumerate(self.models)}

        for domain in self.domains:
            print("Working on domain: {}".format(domain))
            logging.info("Working on domain: {}".format(domain))

            # keep track of evaluated models in this dictionary
            frames = defaultdict(list)

            # we repeat holdout num times
            for i in range(num):
                logging.info("-- holdout iteration: {}".format(i))
                print("-- holdout iteration: {}".format(i))
                self.train_sources, self.test_sources = self._contruct_holdout_samples(domain, holdout)
                logging.info("----> {} test sources: {}".format(len(self.test_sources), self.test_sources))

                for idx, model in model_lookup.items():
                    logging.info("--> evaluating model: {}".format(model))
                    predicted_df = self._evaluate_model(model)
                    if predicted_df is not None:
                        logging.info("---- appending frame")
                        frames[idx].append(predicted_df)

            for idx, model in model_lookup.items():
                logging.info("Concatenating performance frames per model")
                if len(frames[idx]) > 0:
                    predicted_df = pd.concat(frames[idx])
                    if self.debug_csv:
                        if model.debug_csv:
                            debug_csv = model.debug_csv + "_" + domain + ".csv"
                        else:
                            debug_csv = self.debug_csv + "_" + domain + ".csv"
                        predicted_df.to_csv(debug_csv, index=False, header=True, mode="w+")

                    performance = model.evaluate(predicted_df)
                    performance["train_run_time"] = predicted_df["train_run_time"].mean()
                    performance["experiment"] = self.experiment_type
                    performance["experiment_description"] = self.description
                    performance["predict_run_time"] = predicted_df["running_time"].mean()
                    performance["domain"] = domain
                    if len(frames[idx]) == num:
                        performance["status"] = "success"
                    else:
                        performance["status"] = "completed_" + str(len(frames[idx]))
                    performance["model"] = model.model_type
                    performance["model_description"] = model.description
                else:
                    res = {"model": model.model_type,
                           "model_description": model.description,
                           "experiment": self.experiment_type,
                           "experiment_description": self.description,
                           "status": "failure",
                           "domain": domain}
                    performance = pd.DataFrame(res, index=[0])
                performance_frames.append(performance)
                # print("performance: ", performance)

        logging.info("Concatenating performance frames for all models")
        performance = pd.concat(performance_frames, ignore_index=True)
        if self.performance_csv:
            if os.path.exists(self.performance_csv):
                performance.to_csv(self.performance_csv, index=False, header=False, mode="a")
            else:
                performance.to_csv(self.performance_csv, index=False, header=True, mode="w+")

        return True

    def run(self, ):
        """
        Execute experiment
        :return:
        """
        if len(self.domains) < 1:
            logging.warning("Experiment is not possible: there are no domains.")
            print("Experiment is not possible: there are no domains.")
            return False
        if self.experiment_type == "leave_one_out":
            self._leave_one_out()
            return True
        elif self.experiment_type == "repeated_holdout":
            self._repeated_holdout(self.holdout, self.num)
            return True
        else:
            logging.warning("Unsupported experiment type!!!")
            print("Unsupported experiment type!!!")
            return False
