"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Abstract model for semantic labelling/typing.
Evaluation will be based on this abstract model.
"""
import logging
import os

import pandas as pd


class Experiment(object):
    """

    """
    domains = ["soccer", "dbpedia", "museum", "weather", "weapons"]
    # domains = ["weapons"]
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

    def __init__(self, models, experiment_type, description, result_csv, debug_csv):
        """
        Initialize experiment. To run the experiment, please call "run" explicitly.
        :param models:
        :param experiment_type:
        :param description:
        :param result_csv:
        :param debug_csv:
        """
        logging.info("Initializing experiment...")
        self.models = models
        self.experiment_type = experiment_type
        self.description = description
        self.debug_csv = debug_csv
        self.performance_csv = result_csv
        self.train_sources = None
        self.test_sources = None

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
                logging.info("Experiment evaulate: Prediction done.")
                predicted_df1["experiment"] = self.experiment_type
                predicted_df1["experiment_description"] = self.description
                predicted_df1["train_run_time"] = run_time
                frames.append(predicted_df1)
            predicted_df = pd.concat(frames, ignore_index=True)
            return predicted_df
        except Exception as e:
            logging.warning("Model evaluation failed: {}".format(e))
            logging.warning("Model evaluation failed: {}".format(e.args))
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

    def run(self):
        """
        Execute experiment
        :return:
        """
        if self.experiment_type == "leave_one_out":
            self._leave_one_out()
            return True
        else:
            logging.warning("Unsupported experiment type!!!")
            return False
