from unittest import TestCase
from karmaDSL import KarmaSession
from serene_benchmark import Experiment, KarmaDSLModel

class TestSemanticTyper(TestCase):
    def test_predictColumn(self):
        """Verify that DSL server can handle unicode data"""

        dsl = KarmaSession(host="localhost", port=8000)
        print(dsl.ftu())
        vals = [u"Klüft skräms inför på fédéral électoral große", "djfldfjld", u"Эфлолдал"]
        column_predict = dsl.predict_column("test", "test_header", vals)
        print(column_predict)

        self.assertTrue(len(column_predict["predictions"]))
