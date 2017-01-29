from benchmark import Experiment, DINTModel
from serene.core import SchemaMatcher
import os
import logging


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
    resampling_strategy = "NoResampling"
    dint_model = DINTModel(dm, full_feature_config,
                           resampling_strategy,
                           "DINTModel with full feature config and no resampling and filtered types and no parallel",
                           debug_csv=os.path.join("data","debug_dint_full_noresampling_no.csv"))

    # models for experiments
    models = [dint_model]
    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('data', "performance_noresampling_filter_no.csv"),
                                debug_csv=os.path.join("data", "debug_noresampling.csv"))
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
                           debug_csv=os.path.join("data", "debug_dint_full_resampletomean_no.csv"))

    # models for experiments
    models = [dint_model]
    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('data', "performance_resample_filter_no.csv"),
                                debug_csv=os.path.join("data", "debug_resample.csv"))
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
                           debug_csv=os.path.join("data","debug_dint_single_no.csv"))

    models = [dint_model]

    loo_experiment = Experiment(models,
                                experiment_type="leave_one_out",
                                description="plain loo",
                                result_csv=os.path.join('data', "performance_bagging_filter_no.csv"),
                                debug_csv=os.path.join("data", "debug_bagging.csv"))

    loo_experiment.run()

if __name__ == "__main__":

    # setting up the logging
    log_file = 'benchmark.log'
    logging.basicConfig(filename=os.path.join('data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    test_fullfeature_noresampling()
    test_fullfeature_resampletomean()
    test_singlefeatures()

