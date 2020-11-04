import json

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor


@Predictor.register('chid')
class ChIDPredictor(Predictor):
    def load_line(self, line: str) -> JsonDict:
        example = json.loads(line)

        content = example['content']
        candidates = example['candidates']
        real_count = example.get('realCount', 1)
        ground_truths = example.get('groundTruth')

        yield from self._dataset_reader.text_to_instance(
            content,
            candidates,
            ground_truths=ground_truths,
            real_count=real_count
        )

    def dump_line(self, outputs: JsonDict) -> str:
        return json.dumps(outputs, ensure_ascii=False) + '\n'

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        yield from self._dataset_reader.text_to_instance(
            json_dict['content'],
            json_dict['candidates'],
            ground_truths=json_dict.get('ground_truth'),
            real_count=json_dict.get('realCount', 1)
        )
