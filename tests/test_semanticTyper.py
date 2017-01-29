from unittest import TestCase
from benchmark.semantic_typer import SemanticTyper
import pandas as pd

class TestSemanticTyper(TestCase):
    def test_evaluate(self):
        """Verify that metrics are calculated correctly."""
        st = SemanticTyper("test semantic typer", description="test", debug_csv=None)
        predicted_df = pd.DataFrame([{"column_id": 1,
                                     "user_label": 'ax',
                                     "label": 'bx',
                                     "scores_ax": 0.3,
                                     "scores_bx": 0.7,
                                     "scores_unknown": 0},
                                    {"column_id": 2,
                                     "user_label": 'ax',
                                     "label": 'ax',
                                     "scores_ax": 0.7,
                                     "scores_bx": 0.3,
                                     "scores_unknown": 0},
                                     {"column_id": 3,
                                      "user_label": 'ax',
                                      "label": 'unknown',
                                      "scores_ax": 0.35,
                                      "scores_bx": 0.25,
                                      "scores_unknown": 0.4}
                                     ]
                                    )
        print(predicted_df)
        performance = st.evaluate(predicted_df)
        print("--------------result")
        print(performance)
        self.assertEqual(performance["MRR"].values, 2.0/3)
        self.assertEqual(performance["categorical_accuracy"].values, 0.33333333333333331)

    def test_evaluate_binary(self):
        """Verify that metrics are calculated correctly for the binary situation."""
        st = SemanticTyper("test semantic typer", description="test", debug_csv=None)
        predicted_df = pd.DataFrame([{"column_id": 1,
                                     "user_label": 'ax',
                                     "label": 'bx',
                                     "scores_ax": 0.3,
                                     "scores_bx": 0.7,
                                     "scores_unknown": 0},
                                    {"column_id": 2,
                                     "user_label": 'ax',
                                     "label": 'ax',
                                     "scores_ax": 0.7,
                                     "scores_bx": 0.3,
                                     "scores_unknown": 0}
                                     ]
                                    )
        print(predicted_df)
        performance = st.evaluate(predicted_df)
        print("--------------result")
        print(performance)
        self.assertEqual(performance["MRR"].values, 0.75)
        self.assertEqual(performance["categorical_accuracy"].values, 0.5)
