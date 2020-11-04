import os
import pathlib
import shutil
import tempfile

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.testing.model_test_case import ModelTestCase as AllenNlpModelTestCase

__all__ = ["TestCase", "ModelTestCase"]

TEST_DIR = tempfile.mkdtemp(prefix="chid.gnn")


class TestCase(AllenNlpTestCase):
    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / "..").resolve()
    DATA_ROOT = PROJECT_ROOT / "data"
    TRAIN_DATASET = str(DATA_ROOT / "ChID" / "train_data.txt")
    DEV_DATASET = str(DATA_ROOT / "ChID" / "dev_data.txt")
    IDIOM_VECTOR_PATH = str(DATA_ROOT / "idiom_vector.txt")
    IDIOM_DEFINITION_PATH = str(DATA_ROOT / "idiom_definition.json")

    PRETRAINED_ROOT = DATA_ROOT / "pretrained"
    PRETRAINED_MODEL = str(PRETRAINED_ROOT / "chinese_wwm_pytorch")

    MODULE_ROOT = PROJECT_ROOT / "src"
    TOOLS_ROOT = MODULE_ROOT / "tools"
    TESTS_ROOT = MODULE_ROOT / "tests"
    FIXTURES_ROOT = TESTS_ROOT / "fixtures"
    TEST_DIR = pathlib.Path(TEST_DIR)

    def setUp(self):
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR, ignore_errors=True)


class ModelTestCase(TestCase, AllenNlpModelTestCase):
    pass
