"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

"""

from serene_benchmark import Experiment, DINTModel, NNetModel
from serene.matcher.core import SchemaMatcher
import os
import logging
from logging.handlers import RotatingFileHandler


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
                                                       "min-editdistqance-from-class-examples"]
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
                           ignore_unknown=ignore_uknown,
                           num_bags=100, bag_size=100)
    return dint_model


def test_simple(ignore_uknown=True, domains=None):
    # ******* setting up DINTModel
    dm = SchemaMatcher(host="localhost", port=8080)
    # dictionary with features

    single_feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                                "ratio-alpha-chars", "prop-numerical-chars", "shannon-entropy",
                                                "prop-whitespace-chars", "prop-entries-with-at-sign"]
                             }
    # resampling strategy
    resampling_strategy = "NoResampling"
    dint_model = DINTModel(dm, single_feature_config,
                           resampling_strategy,
                           "DINTModel with simple feature config",
                           debug_csv=os.path.join("results","debug_dint_simple.csv"),
                           ignore_unknown=ignore_uknown)

    models = [dint_model]

    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_simple.csv"),
                                debug_csv=os.path.join("results", "debug_simple.csv"))

    weapons = ["www.theoutdoorstrader.com.csv", "www.tennesseegunexchange.com.csv",
                    "www.montanagunclassifieds.com.csv", "www.kyclassifieds.com.csv",
                    "www.hawaiiguntrader.com.csv", "www.gunsinternational.com.csv",
                    "www.floridaguntrader.com.csv", "www.floridagunclassifieds.com.csv",
                    "www.elpasoguntrader.com.csv", "www.dallasguns.com.csv",
                    "www.armslist.com.csv", "www.alaskaslist.com.csv"
                    ]

    if domains:
        loo_experiment.change_domains(domains)
    loo_experiment.run()


def test_simple_holdout():
    # ******* setting up DINTModel
    dm = SchemaMatcher(host="localhost", port=8080)
    # dictionary with features

    single_feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                                "ratio-alpha-chars", "prop-numerical-chars",
                                                "prop-whitespace-chars", "prop-entries-with-at-sign"]
                             }
    # resampling strategy
    resampling_strategy = "NoResampling"
    dint_model = DINTModel(dm, single_feature_config,
                           resampling_strategy,
                           "DINTModel with simple feature config",
                           debug_csv=os.path.join("results","debug_dint_simple_holdout.csv"))

    models = [dint_model]

    loo_experiment = Experiment(models,
                                experiment_type="repeated_holdout",
                                description="repeated_holdout_0.5_2",
                                result_csv=os.path.join('results', "performance_simple_holdout.csv"),
                                debug_csv=os.path.join("results", "debug_simple_holdout.csv"),
                                holdout=0.5,
                                num=2)

    loo_experiment.run()


def test_city(dm):
    # dictionary with features
    full_feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
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
                                                       "num-neighbours": 5
                                                       }, {"name": "min-wordnet-jcn-distance-from-class-examples",
                                                           "max-comparisons-per-class": 5
                                                           },
                                                      {"name": "min-wordnet-lin-distance-from-class-examples",
                                                       "max-comparisons-per-class": 5
                                                       }]
                           }

    # resampling strategy
    resampling_strategy = "Bagging"
    dint_model = DINTModel(dm, full_feature_config,
                           resampling_strategy,
                           "DINTModel: resampling {}, features fullchardist".format(resampling_strategy),
                           debug_csv=os.path.join("results","debug_dint_chardist_noresampling_head.csv"))

    # models for experiments
    models = [dint_model]
    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_noresampling_chardist.csv"),
                                debug_csv=os.path.join("results", "debug_noresampling_head.csv"))
    loo_experiment.change_domains(domains=["dbpedia"])
    loo_experiment.run()


def test_noheaderfeature_noresampling():
    # ******* setting up DINTModel
    dm = SchemaMatcher(host="localhost", port=8080)
    # dictionary with features
    full_feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                              "ratio-alpha-chars", "prop-numerical-chars",
                                              "prop-whitespace-chars", "prop-entries-with-at-sign",
                                              "prop-entries-with-hyphen", "prop-entries-with-paren",
                                              "prop-entries-with-currency-symbol", "mean-commas-per-entry",
                                              "mean-forward-slashes-per-entry",
                                              "prop-range-format", "is-discrete", "entropy-for-discrete-values"],
                           "activeFeatureGroups": ["stats-of-text-length", "stats-of-numerical-type",
                                                   "prop-instances-per-class-in-knearestneighbours",
                                                   "mean-character-cosine-similarity-from-class-examples"],
                           "featureExtractorParams": [{"name": "prop-instances-per-class-in-knearestneighbours",
                                                       "num-neighbours": 5
                                                       }, {"name": "min-wordnet-jcn-distance-from-class-examples",
                                                           "max-comparisons-per-class": 5
                                                           },
                                                      {"name": "min-wordnet-lin-distance-from-class-examples",
                                                       "max-comparisons-per-class": 5
                                                       }]
                           }

    # resampling strategy
    resampling_strategy = "BaggingToMax"
    dint_model = DINTModel(dm, full_feature_config,
                           resampling_strategy,
                           "DINTModel with feature config without headers and baggingtomax and headers changed",
                           debug_csv=os.path.join("results","debug_dint_noheader_baggingtomax.csv"))

    # models for experiments
    models = [dint_model]
    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_baggingtomax_noheader.csv"),
                                debug_csv=os.path.join("results", "debug_noheader.csv"))
    loo_experiment.run()

def test_fullfeature_resampletomean():
    # ******* setting up DINTModel
    dm = SchemaMatcher(host="localhost", port=8080)
    # dictionary with features

    full_feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                              "ratio-alpha-chars", "prop-numerical-chars",
                                              "prop-whitespace-chars", "prop-entries-with-at-sign",
                                              "prop-entries-with-hyphen", "prop-entries-with-paren",
                                              "prop-entries-with-currency-symbol", "mean-commas-per-entry",
                                              "mean-forward-slashes-per-entry",
                                              "prop-range-format", "is-discrete", "entropy-for-discrete-values"],
                           "activeFeatureGroups": ["stats-of-text-length", "stats-of-numerical-type",
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

    # resampling strategy
    resampling_strategy = "ResampleToMean"
    dint_model = DINTModel(dm, full_feature_config,
                           resampling_strategy,
                           "DINTModel with full feature config and resampleToMean  and filtered types and no parallel",
                           debug_csv=os.path.join("results", "debug_dint_full_resampletomean_no.csv"))

    # models for experiments
    models = [dint_model]
    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_resample_filter_no.csv"),
                                debug_csv=os.path.join("results", "debug_resample.csv"))
    loo_experiment.run()


def test_fullfeature_bagging():
    # ******* setting up DINTModel
    dm = SchemaMatcher(host="localhost", port=8080)
    # dictionary with features

    full_feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                              "ratio-alpha-chars", "prop-numerical-chars",
                                              "prop-whitespace-chars", "prop-entries-with-at-sign",
                                              "prop-entries-with-hyphen", "prop-entries-with-paren",
                                              "prop-entries-with-currency-symbol", "mean-commas-per-entry",
                                              "mean-forward-slashes-per-entry",
                                              "prop-range-format", "is-discrete", "entropy-for-discrete-values"],
                           "activeFeatureGroups": ["stats-of-text-length", "stats-of-numerical-type",
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

    # resampling strategy
    resampling_strategy = "BaggingToMax"
    dint_model = DINTModel(dm, full_feature_config,
                           resampling_strategy,
                           "DINTModel with full feature config and baggingtomax and filtered types and changed headers",
                           debug_csv=os.path.join("results", "debug_dint_head_baggingtomax.csv"))

    # models for experiments
    models = [dint_model]
    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_full_baggingtomax_head.csv"),
                                debug_csv=os.path.join("results", "debug_baggingtomax.csv"))
    loo_experiment.run()


def test_singlefeatures():
    # ******* setting up DINTModel
    dm = SchemaMatcher(host="localhost", port=8080)
    # dictionary with features

    single_feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                                "ratio-alpha-chars", "prop-numerical-chars",
                                                "prop-whitespace-chars", "prop-entries-with-at-sign",
                                                "prop-entries-with-hyphen", "prop-entries-with-paren",
                                                "prop-entries-with-currency-symbol", "mean-commas-per-entry",
                                                "mean-forward-slashes-per-entry",
                                                "prop-range-format", "is-discrete", "entropy-for-discrete-values"]
                             }
    # resampling strategy
    resampling_strategy = "BaggingToMax"
    dint_model = DINTModel(dm, single_feature_config,
                           resampling_strategy,
                           "DINTModel with single feature config and baggingtomax  and filtered types and no parallel",
                           debug_csv=os.path.join("results","debug_dint_single_no.csv"))

    models = [dint_model]

    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_bagging_filter_no.csv"),
                                debug_csv=os.path.join("results", "debug_bagging.csv"))

    loo_experiment.run()


def test_models():
    # ******* setting up DINTModel
    dm = SchemaMatcher(host="localhost", port=8080)

    logging.info("Cleaning models from DINT server")
    for m in dm.models:
        dm.remove_model(m)
    logging.info("Cleaning datasets from DINT server")
    for ds in dm.datasets:
        dm.remove_dataset(ds)

    m1 = create_dint_model(dm, "full", "NoResampling")
    m2 = create_dint_model(dm, "single", "NoResampling")
    m3 = create_dint_model(dm, "chardist", "NoResampling")
    m4 = create_dint_model(dm, "noheader", "NoResampling")
    m5 = create_dint_model(dm, "chardistonly", "NoResampling")
    rf_model = NNetModel(['rf@charfreq'], 'rf@charfreq model: no headers', add_headers=False, p_header=0,
                         debug_csv=os.path.join("results", "debug_nnet_rf.csv"))

    models = [m1, m2, m3, m4, m5, rf_model]

    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_models_loo.csv"),
                                debug_csv=os.path.join("results", "debug_models_loo.csv"))

    loo_experiment.run()


def test_models_holdout():
    # ******* setting up DINTModel
    dm = SchemaMatcher(host="localhost", port=8080)

    logging.info("Cleaning models from DINT server")
    for m in dm.models:
        dm.remove_model(m)
    logging.info("Cleaning datasets from DINT server")
    for ds in dm.datasets:
        dm.remove_dataset(ds)

    m1 = create_dint_model(dm, "full", "NoResampling")
    m2 = create_dint_model(dm, "single", "NoResampling")
    m3 = create_dint_model(dm, "chardist", "NoResampling")
    m4 = create_dint_model(dm, "noheader", "NoResampling")
    m5 = create_dint_model(dm, "chardistonly", "NoResampling")
    rf_model = NNetModel(['rf@charfreq'], 'rf@charfreq model: no headers', add_headers=False, p_header=0,
                         debug_csv=os.path.join("results", "debug_nnet_rf_holdout.csv"))

    models = [m1, m2, m3, m4, m5, rf_model]

    rhold_experiment = Experiment(models,
                                experiment_type="repeated_holdout",
                                description="repeated_holdout_0.5_10",
                                result_csv=os.path.join('results', "performance_models_holdout.csv"),
                                debug_csv=os.path.join("results", "debug_holdout.csv"),
                                holdout=0.5,
                                num=10)

    rhold_experiment.run()

def test_bagging():
    # ******* setting up DINTModel
    dm = SchemaMatcher(host="localhost", port=8080)

    logging.info("Cleaning models from DINT server")
    for m in dm.models:
        dm.remove_model(m)
    logging.info("Cleaning datasets from DINT server")
    for ds in dm.datasets:
        dm.remove_dataset(ds)

    m1 = create_dint_model(dm, "full", "Bagging")
    m2 = create_dint_model(dm, "single", "Bagging")
    m3 = create_dint_model(dm, "chardist", "Bagging")
    m4 = create_dint_model(dm, "noheader", "Bagging")
    m5 = create_dint_model(dm, "chardistonly", "Bagging")

    models = [m1, m2, m3, m4, m5]

    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_dint_bagging.csv"),
                                debug_csv=os.path.join("results", "debug_dint.csv"))

    loo_experiment.run()


def test_upsampletomax():
    # ******* setting up DINTModel
    dm = SchemaMatcher(host="localhost", port=8080)

    logging.info("Cleaning models from DINT server")
    for m in dm.models:
        dm.remove_model(m)
    logging.info("Cleaning datasets from DINT server")
    for ds in dm.datasets:
        dm.remove_dataset(ds)

    m1 = create_dint_model(dm, "full", "UpsampleToMax")
    m2 = create_dint_model(dm, "single", "UpsampleToMax")
    m3 = create_dint_model(dm, "full_chardist", "UpsampleToMax")
    m4 = create_dint_model(dm, "noheader", "UpsampleToMax")
    m5 = create_dint_model(dm, "chardistonly", "UpsampleToMax")

    models = [m1, m2, m3, m4, m5]

    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_dint_upsample.csv"),
                                debug_csv=os.path.join("results", "debug_dint_upsample.csv"))

    loo_experiment.run()

def test_resampletomean():
    # ******* setting up DINTModel
    dm = SchemaMatcher(host="localhost", port=8080)

    logging.info("Cleaning models from DINT server")
    for m in dm.models:
        dm.remove_model(m)
    logging.info("Cleaning datasets from DINT server")
    for ds in dm.datasets:
        dm.remove_dataset(ds)

    m1 = create_dint_model(dm, "full", "ResampleToMean")
    m2 = create_dint_model(dm, "single", "ResampleToMean")
    m3 = create_dint_model(dm, "full_chardist", "ResampleToMean")
    m4 = create_dint_model(dm, "noheader", "ResampleToMean")
    m5 = create_dint_model(dm, "chardistonly", "ResampleToMean")

    models = [m1, m2, m3, m4, m5]

    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_dint_resampletomean.csv"),
                                debug_csv=os.path.join("results", "debug_dint_resampletomean.csv"))

    loo_experiment.run()


def make_experiment(dm_session, strategies, cur_experiments, cur_features, unknown_ignore, domains=None):
    for strat in strategies:
        print("Picking strategy: ", strat)
        for exp in cur_experiments:
            print("Performing experiment:", exp)
            for ig in unknown_ignore:
                print("Setting ignore_unknown: ", ig)
                models = [create_dint_model(dm_session, feat, strat, ig) for feat in cur_features]
                experiment = Experiment(models,
                                        experiment_type=exp,
                                        description=exp + "_ignore" + str(ig) + "_strategy" + strat,
                                        result_csv=os.path.join('results', "performance_dint_{}_ignore{}.csv".format(
                                            strat, ig)
                                                                ),
                                        debug_csv=os.path.join("results", "debug_dint_{}_ignore{}.csv".format(
                                            strat, ig)
                                                               ),
                                        holdout=0.2, num=10)

                if domains:
                    experiment.change_domains(domains)
                experiment.run()


def benchmark_bagging_num(dm_session):
    feature_config = {"activeFeatures": ["shannon-entropy"],
                      "activeFeatureGroups": ["char-dist-features"]}
    print("Picking strategy: Bagging")
    print("Performing experiment: repeated_holdout", )
    print("Setting ignore_unknown: True")

    d1 = DINTModel(dm_session, feature_config,
                   "Bagging",
                   "DINTModel: resampling Bagging, chardistonly",
                   debug_csv=os.path.join("results", "debug_dint_Bagging_chardist.csv"),
                   ignore_unknown=True,
                   num_bags=100, bag_size=100)
    d2 = DINTModel(dm_session, feature_config,
                   "Bagging",
                   "DINTModel: resampling Bagging, chardistonly",
                   debug_csv=os.path.join("results", "debug_dint_Bagging_chardist.csv"),
                   ignore_unknown=True,
                   num_bags=10, bag_size=100)
    d3 = DINTModel(dm_session, feature_config,
                   "Bagging",
                   "DINTModel: resampling Bagging, chardistonly",
                   debug_csv=os.path.join("results", "debug_dint_Bagging_chardist.csv"),
                   ignore_unknown=True,
                   num_bags=150, bag_size=100)
    d4 = DINTModel(dm_session, feature_config,
                   "Bagging",
                   "DINTModel: resampling Bagging, chardistonly",
                   debug_csv=os.path.join("results", "debug_dint_Bagging_chardist.csv"),
                   ignore_unknown=True,
                   num_bags=50, bag_size=100)

    experiment = Experiment([d1,d2,d3,d4],
                            experiment_type="repeated_holdout",
                            description="repeated_holdout_ignoreTrue_strategyBagging",
                            result_csv=os.path.join('results', "performance_dint_Bagging_ignoreTrue.csv"),
                            debug_csv=os.path.join("results", "debug_dint_Bagging_ignoreTrue.csv"),
                            holdout=0.2, num=10)
    experiment.run()

    # experiment = Experiment([d1, d2, d3, d4],
    #                         experiment_type="leave_one_out",
    #                         description="leave_one_out_ignoreFalse_strategyBagging",
    #                         result_csv=os.path.join('results', "performance_dint_Bagging_ignoreFalse.csv"),
    #                         debug_csv=os.path.join("results", "debug_dint_Bagging_ignoreFalse.csv"),
    #                         holdout=0.2, num=10)
    # experiment.run()

def benchmark_bagging_size(dm_session):
    feature_config = {"activeFeatures": ["shannon-entropy"],
                      "activeFeatureGroups": ["char-dist-features"]}
    print("Picking strategy: Bagging")
    print("Performing experiment: repeated_holdout", )
    print("Setting ignore_unknown: True")

    d1 = DINTModel(dm_session, feature_config,
                   "Bagging",
                   "DINTModel: resampling Bagging, chardistonly",
                   debug_csv=os.path.join("results", "debug_dint_Bagging_chardist_ignoreTrue.csv"),
                   ignore_unknown=True,
                   num_bags=50, bag_size=10)
    d2 = DINTModel(dm_session, feature_config,
                   "Bagging",
                   "DINTModel: resampling Bagging, chardistonly",
                   debug_csv=os.path.join("results", "debug_dint_Bagging_chardist_ignoreTrue.csv"),
                   ignore_unknown=True,
                   num_bags=50, bag_size=30)
    d3 = DINTModel(dm_session, feature_config,
                   "Bagging",
                   "DINTModel: resampling Bagging, chardistonly",
                   debug_csv=os.path.join("results", "debug_dint_Bagging_chardist_ignoreTrue.csv"),
                   ignore_unknown=True,
                   num_bags=50, bag_size=50)
    # d4 = DINTModel(dm_session, feature_config,
    #                "Bagging",
    #                "DINTModel: resampling Bagging, chardistonly",
    #                debug_csv=os.path.join("results", "debug_dint_Bagging_chardist_ignoreTrue.csv"),
    #                ignore_unknown=True,
    #                num_bags=50, bag_size=80)
    d5 = DINTModel(dm_session, feature_config,
                   "Bagging",
                   "DINTModel: resampling Bagging, chardistonly",
                   debug_csv=os.path.join("results", "debug_dint_Bagging_chardist_ignoreTrue.csv"),
                   ignore_unknown=True,
                   num_bags=50, bag_size=100)

    experiment = Experiment([d1,d2,d3,d5],
                            experiment_type="repeated_holdout",
                            description="repeated_holdout_ignoreTrue_strategyBagging",
                            result_csv=os.path.join('results', "performance_dint_Bagging_ignoreTrue.csv"),
                            debug_csv=os.path.join("results", "debug_dint_Bagging_ignoreTrue.csv"),
                            holdout=0.2, num=3)
    experiment.change_domains(domains=["soccer", "museum","dbpedia","weather", "weapons"])
    experiment.run()


if __name__ == "__main__":

    # setting up the logging
    log_file = os.path.join('results', 'benchmark_dint.log')
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s: %(message)s',
                                      '%Y-%m-%d %H:%M:%S')
    my_handler = RotatingFileHandler(log_file, mode='a', maxBytes=5*1024*1024,
                                 backupCount=5, encoding=None, delay=0)
    my_handler.setFormatter(log_formatter)
    my_handler.setLevel(logging.DEBUG)
    # logging.basicConfig(filename=log_file,
    #                     level=logging.DEBUG, filemode='w+',
    #                     format='%(asctime)s %(levelname)s %(module)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(my_handler)

    # test_simple(ignore_uknown=False, domains=["museum"])
    # test_simple_holdout()
    # test_models()
    # test_models_holdout()
    # test_bagging()
    # test_upsampletomax()
    # test_resampletomean()
    # test_simple(ignore_uknown=True, domains=["weapons"])

    # establish connection to Serene server
    dm = SchemaMatcher(host="localhost", port=8080)
    logging.info("Cleaning models from DINT server")
    # for m in dm.models:
    #     dm.remove_model(m)
    # logging.info("Cleaning datasets from DINT server")
    # for ds in dm.datasets:
    #     dm.remove_dataset(ds)

    ######################################
    # all will be run only for no resampling
    # features = ["fullchardist"]
    # experiments = ["repeated_holdout"]
    # make_experiment(dm_session=dm, strategies=["NoResampling"],
    #                 cur_experiments=experiments, cur_features=features,
    #                 unknown_ignore=[True])

    # make_experiment(dm_session=dm, strategies=["NoResampling"],
    #                 cur_experiments=experiments, cur_features=features,
    #                 unknown_ignore=[False], domains=["soccer"])

    ######################################
    resampling_strategies = ["Bagging", "ResampleToMean", "NoResampling"]
    resampling_strategies = ["NoResampling"]
    features = ["chardistonly", "fullcity", "fullchardist", "chardist-edit"]
    experiments = ["leave_one_out"]

    # features = ["chardist-edit","fullcity"]sys
    # make_experiment(dm_session=dm, strategies=resampling_strategies,
    #                 cur_experiments=experiments, cur_features=features,
    #                 unknown_ignore=[True], domains=["dbpedia"])

    # features = ["fullchardist"]
    # make_experiment(dm_session=dm, strategies=resampling_strategies,
    #                 cur_experiments=experiments, cur_features=features,
    #                 unknown_ignore=[True])

    resampling_strategies = ["Bagging", "ResampleToMean", "NoResampling"]
    features = ["fullchardist", "chardist-edit"]
    make_experiment(dm_session=dm, strategies=resampling_strategies,
                    cur_experiments=experiments, cur_features=features,
                    unknown_ignore=[False], domains=["museum"])


    ####################################
    # test_city(dm)
    # benchmark_bagging_num(dm)
    # benchmark_bagging_size(dm)