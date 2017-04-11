from serene_benchmark import Experiment, DINTModel
from serene.core import SchemaMatcher
import os
import logging



def create_dint_model(dm, features="full", resampling_strategy="NoResampling"):
    """
    Create dint model with specified parameters
    :param dm: SchemaMatcher
    :param features: string ("full", "single", "noheader", "chardist")
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
                                                "prop-range-format", "is-discrete", "entropy-for-discrete-values"]
                      }
    if features == "full":
        feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                              "ratio-alpha-chars", "prop-numerical-chars",
                                              "prop-whitespace-chars", "prop-entries-with-at-sign",
                                              "prop-entries-with-hyphen", "prop-entries-with-paren",
                                              "prop-entries-with-currency-symbol", "mean-commas-per-entry",
                                              "mean-forward-slashes-per-entry",
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
    elif features=="chardist":
        feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                                  "ratio-alpha-chars", "prop-numerical-chars",
                                                  "prop-whitespace-chars", "prop-entries-with-at-sign",
                                                  "prop-entries-with-hyphen", "prop-entries-with-paren",
                                                  "prop-entries-with-currency-symbol", "mean-commas-per-entry",
                                                  "mean-forward-slashes-per-entry",
                                                  "prop-range-format", "is-discrete", "entropy-for-discrete-values"],
                               "activeFeatureGroups": ["char-dist-features", "stats-of-text-length",
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
    elif features=="chardistonly":
        feature_config = {"activeFeatures": ["entropy-for-discrete-values"],
                               "activeFeatureGroups": ["char-dist-features"]}
    elif features=="noheader":
        feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals",
                                                  "ratio-alpha-chars", "prop-numerical-chars",
                                                  "prop-whitespace-chars", "prop-entries-with-at-sign",
                                                  "prop-entries-with-hyphen", "prop-entries-with-paren",
                                                  "prop-entries-with-currency-symbol", "mean-commas-per-entry",
                                                  "mean-forward-slashes-per-entry",
                                                  "prop-range-format", "is-discrete", "entropy-for-discrete-values"],
                               "activeFeatureGroups": ["char-dist-features", "stats-of-text-length",
                                                       "stats-of-numerical-type",
                                                       "mean-character-cosine-similarity-from-class-examples"]
                               }

    dint_model = DINTModel(dm, feature_config,
                           resampling_strategy,
                           "DINTModel: resampling {}, features {}".format(resampling_strategy, features),
                           debug_csv=os.path.join("results",
                                                  "debug_dint_{}_{}.csv".format(resampling_strategy, features)))
    return dint_model


def test_fullfeature_noresampling():
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
                           "activeFeatureGroups": ["char-dist-features", "stats-of-text-length",
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

    # resampling strategy
    resampling_strategy = "NoResampling"
    dint_model = DINTModel(dm, full_feature_config,
                           resampling_strategy,
                           "DINTModel with chardist and no resampling and changed headers",
                           debug_csv=os.path.join("results","debug_dint_chardist_noresampling_head.csv"))

    # models for experiments
    models = [dint_model]
    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_noresampling_chardist.csv"),
                                debug_csv=os.path.join("results", "debug_noresampling_head.csv"))
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

    models = [m1, m2, m3, m4, m5]

    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('results', "performance_dint.csv"),
                                debug_csv=os.path.join("results", "debug_dint.csv"))

    loo_experiment.run()

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



if __name__ == "__main__":

    # setting up the logging
    log_file = 'benchmark_dint.log'
    logging.basicConfig(filename=os.path.join('results', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    test_upsampletomax()
    test_resampletomean()
    test_models()
    test_bagging()
