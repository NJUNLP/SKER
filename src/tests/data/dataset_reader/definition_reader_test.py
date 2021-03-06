from allennlp.common.util import ensure_list
from allennlp.data.token_indexers import PretrainedBertIndexer

from src.common import TestCase
from src.data.dataset_reader import DefinitionReader


class TestDefinitionReader(TestCase):
    def test_reader_from_file(self):
        reader = DefinitionReader(
            idiom_vector_path=self.IDIOM_VECTOR_PATH,
            idiom_definition_path=self.IDIOM_DEFINITION_PATH,
            pretrained_model=self.PRETRAINED_MODEL,
            content_token_indexer={
                "bert": PretrainedBertIndexer(
                    self.PRETRAINED_MODEL
                )
            },
            max_seq_length=256
        )
        instances = reader.read(str(self.FIXTURES_ROOT / "data" / "realcount_3_sample.txt"))
        instances = ensure_list(instances)

        assert len(instances) == 3
