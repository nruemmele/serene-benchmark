"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

"""

from serene_benchmark import Experiment, KarmaDSLModel, NNetModel, DINTModel
import os
from karmaDSL import KarmaSession
import logging
from logging.handlers import RotatingFileHandler
from serene.matcher.core import SchemaMatcher

def create_dint_model(dm, features="full", resampling_strategy="NoResampling", ignore_uknown=True):
    """
    Create dint model with specified parameters
    :param dm: SchemaMatcher
    :param features: string ("full", "single", "noheader", "fullchardist", "chardistonly","chardist-rfknn")
    :param resampling_strategy: resampling strategy, default None
    :return:
    """
    # dictionary with features
    feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                         "ratio-alpha-chars", "prop-numerical-chars",
                                         "prop-whitespace-chars", "prop-entries-with-at-sign",
                                         "prop-entries-with-hyphen", "prop-entries-with-paren",
                                         "prop-entries-with-currency-symbol", "mean-commas-per-entry",
                                         "mean-forward-slashes-per-entry",
                                         "prop-range-format", "is-discrete",
                                         "entropy-for-discrete-values", "shannon-entropy"]
                      }
    if features == "full":
        feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                              "ratio-alpha-chars", "prop-numerical-chars",
                                              "prop-whitespace-chars", "prop-entries-with-at-sign",
                                              "prop-entries-with-hyphen", "prop-entries-with-paren",
                                              "prop-entries-with-currency-symbol", "mean-commas-per-entry",
                                              "mean-forward-slashes-per-entry", "shannon-entropy",
                                              "prop-range-format", "is-discrete", "entropy-for-discrete-values"],
                           "activeFeatureGroups": ["stats-of-text-length",
                                                   "stats-of-numerical-type",
                                                   "prop-instances-per-class-in-knearestneighbours",
                                                   "mean-character-cosine-similarity-from-class-examples",
                                                   "min-editdistance-from-class-examples",
                                                   "min-wordnet-jcn-distance-from-class-examples",
                                                   "min-wordnet-lin-distance-from-class-examples"],
                           "featureExtractorParams": [{"name": "prop-instances-per-class-in-knearestneighbours",
                                                       "num-neighbours": 5
                                                       }, {"name": "min-wordnet-jcn-distance-from-class-examples",
                                                           "max-comparisons-per-class": 5
                                                           },
                                                      {"name": "min-wordnet-lin-distance-from-class-examples",
                                                       "max-comparisons-per-class": 5
                                                       }]
                           }
    elif features=="fullchardist":
        feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                                  "ratio-alpha-chars", "prop-numerical-chars",
                                                  "prop-whitespace-chars", "prop-entries-with-at-sign",
                                                  "prop-entries-with-hyphen", "prop-entries-with-paren",
                                                  "prop-entries-with-currency-symbol", "mean-commas-per-entry",
                                                  "shannon-entropy", "mean-forward-slashes-per-entry",
                                                  "prop-range-format", "is-discrete", "entropy-for-discrete-values"],
                               "activeFeatureGroups": ["char-dist-features", "stats-of-text-length",
                                                       "stats-of-numerical-type",
                                                       "prop-instances-per-class-in-knearestneighbours",
                                                       "mean-character-cosine-similarity-from-class-examples",
                                                       "min-editdistance-from-class-examples",
                                                       "min-wordnet-jcn-distance-from-class-examples",
                                                       "min-wordnet-lin-distance-from-class-examples"],
                               "featureExtractorParams": [{"name": "prop-instances-per-class-in-knearestneighbours",
                                                           "num-neighbours": 3
                                                           }, {"name": "min-wordnet-jcn-distance-from-class-examples",
                                                               "max-comparisons-per-class": 3
                                                               },
                                                          {"name": "min-wordnet-lin-distance-from-class-examples",
                                                           "max-comparisons-per-class": 3
                                                           }]
                               }
    elif features=="chardist-edit":
        feature_config = {"activeFeatures": ["shannon-entropy"],
                               "activeFeatureGroups": ["char-dist-features",
                                                       "min-editdistance-from-class-examples"]
                               }
    elif features=="fullcity":
        feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                                  "ratio-alpha-chars", "prop-numerical-chars",
                                                  "prop-whitespace-chars", "prop-entries-with-at-sign",
                                                  "prop-entries-with-hyphen", "prop-entries-with-paren",
                                                  "prop-entries-with-currency-symbol", "mean-commas-per-entry",
                                                  "mean-forward-slashes-per-entry",
                                                  "prop-range-format", "is-discrete", "entropy-for-discrete-values",
                                                  "shannon-entropy"],
                               "activeFeatureGroups": ["char-dist-features", "stats-of-text-length",
                                                       "stats-of-numerical-type",
                                                       "mean-character-cosine-similarity-from-class-examples"],
                               "featureExtractorParams": [{"name": "prop-instances-per-class-in-knearestneighbours",
                                                           "num-neighbours": 3
                                                           }, {"name": "min-wordnet-jcn-distance-from-class-examples",
                                                               "max-comparisons-per-class": 3
                                                               },
                                                          {"name": "min-wordnet-lin-distance-from-class-examples",
                                                           "max-comparisons-per-class": 3
                                                           }]
                          }
    elif features=="chardistonly":
        feature_config = {"activeFeatures": ["shannon-entropy"],
                               "activeFeatureGroups": ["char-dist-features"]}
    elif features=="noheader":
        feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                                  "ratio-alpha-chars", "prop-numerical-chars",
                                                  "prop-whitespace-chars", "prop-entries-with-at-sign",
                                                  "prop-entries-with-hyphen", "prop-entries-with-paren",
                                                  "prop-entries-with-currency-symbol", "mean-commas-per-entry",
                                                  "shannon-entropy", "mean-forward-slashes-per-entry",
                                                  "prop-range-format", "is-discrete", "entropy-for-discrete-values"],
                               "activeFeatureGroups": ["char-dist-features", "stats-of-text-length",
                                                       "stats-of-numerical-type",
                                                       "mean-character-cosine-similarity-from-class-examples"]
                               }
    elif features=="chardist-rfknn":
        feature_config = {"activeFeatures": ["shannon-entropy"],
                          "activeFeatureGroups": ["char-dist-features",
                                                  "prop-instances-per-class-in-knearestneighbours"],
                          "featureExtractorParams": [{"name": "prop-instances-per-class-in-knearestneighbours",
                                                      "num-neighbours": 5
                                                      }]
                          }
    else:
        features="single"

    dint_model = DINTModel(dm, feature_config,
                           resampling_strategy,
                           "DINTModel: resampling {}, features {}".format(resampling_strategy, features),
                           debug_csv=os.path.join("results",
                                                  "debug_dint_{}_{}_ignore{}.csv".format(
                                                      resampling_strategy, features, ignore_uknown)),
                           ignore_unknown=ignore_uknown)
    return dint_model


def create_dint_models(dm_session, strategies, cur_features, ignore_unknown):
    return [create_dint_model(dm_session, feat, strat, ignore_unknown) for feat in cur_features
              for strat in strategies]


def create_models(dm_session, dsl_session, experiment, ignore, dint_resampling, dint_features):
    dsl_model = KarmaDSLModel(dsl_session, "DSL",
                              ignore_unknown=ignore, prediction_type="column",
                              debug_csv=os.path.join("results",
                                                     "debug_dsl_column_{}_ignore{}.csv".format(
                                                         experiment, ignore)
                                                     ))
    dsl_model_plus = KarmaDSLModel(dsl_session, "DSL+",
                                   ignore_unknown=ignore, prediction_type="enhanced",
                                   debug_csv=os.path.join("results",
                                                          "debug_dsl_plus_column_{}_ignore{}.csv".format(
                                                              experiment, ignore)
                                                          ))
    cnn_model = NNetModel(['cnn@charseq'], 'cnn@charseq model: no headers',
                          add_headers=False, p_header=0,
                          debug_csv=os.path.join("results", "debug_nnet_cnn_ignore{}_{}.csv".format(
                              ignore, experiment)),
                          ignore_unknown=ignore)
    mlp_model = NNetModel(['mlp@charfreq'], 'mlp@charfreq model: no headers',
                          add_headers=False, p_header=0,
                          debug_csv=os.path.join("results", "debug_nnet_mlp_ignore{}_{}.csv".format(
                              ignore, experiment)),
                          ignore_unknown=ignore)

    rf_model = NNetModel(['rf@charfreq'], 'rf@charfreq model: no headers', add_headers=False, p_header=0,
                         debug_csv=os.path.join("results", "debug_nnet_rf_ignore{}_{}.csv".format(
                             ignore, experiment)),
                         ignore_unknown=ignore)

    return [dsl_model, dsl_model_plus, rf_model, mlp_model, cnn_model] + \
             create_dint_models(dm_session, dint_resampling, dint_features, ignore)

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

    ########################## establish connections to the servers
    dsl = KarmaSession(host="localhost", port=8000)
    dm = SchemaMatcher(host="localhost", port=8080)

    resampling_strategies = ["NoResampling", "ResampleToMean", "Bagging"]
    features = ["chardist-edit", "chardistonly", "fullchardist", "fullcity"]

    domains = None
    experiments = ["leave_one_out", "repeated_holdout"]

    ######################ignore unmapped attributes##########################
    print("Setting ignore_unknown: ", True)
    for experiment_type in experiments:

        models = create_models(dm, dsl, experiment_type,
                               True, resampling_strategies, features)

        experiment = Experiment(models,
                                experiment_type=experiment_type,
                                description=experiment_type,
                                result_csv=os.path.join('results',
                                                        "performance_{}_ignore{}.csv".format(
                                                            experiment_type, True)),
                                debug_csv=os.path.join("results", "debug.csv"),
                                holdout=0.2, num=1
                                )

        if domains:
            experiment.change_domains(domains)
        experiment.run()

    ######################unknown attributes##########################
    print("Setting ignore_unknown: ", False)
    for experiment_type in experiments:
        models = create_models(dm, dsl, experiment_type,
                               False, resampling_strategies, features)

        experiment = Experiment(models,
                                experiment_type=experiment_type,
                                description=experiment_type,
                                result_csv=os.path.join('results',
                                                        "performance_{}_ignore{}.csv".format(
                                                            experiment_type, False)),
                                debug_csv=os.path.join("results", "debug.csv"),
                                holdout=0.2, num=10
                                )

        experiment.change_domains(domains=["museum", "soccer"])
        experiment.run()


