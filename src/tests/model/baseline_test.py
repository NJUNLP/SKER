from src.common import ModelTestCase

from src.data.dataset_reader import BaselineReader
from src.model.baseline import Baseline


class BaselineTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            str(self.FIXTURES_ROOT / "config" / "baseline.jsonnet"),
            str(self.FIXTURES_ROOT / "data" / "chid.txt")
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.param_file
        )
