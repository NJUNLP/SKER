import argparse
from collections import Counter
import json

from allennlp.common import Params
from allennlp.data import DatasetReader
from allennlp.training import util as training_util

from src.data.dataset_reader.baseline_reader import BaselineReader


def parse_args():
    parser = argparse.ArgumentParser(
        "Analyze Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "param_path",
        type=str,
        help='path to parameter file describing the model to be trained'
    )
    parser.add_argument(
        '-o', '--overrides',
        type=str,
        default="",
        help='a JSON structure used to override the experiment configuration'
    )
    return parser.parse_args()


def main(param_path, overrides):
    params = Params.from_file(param_path, overrides)
    all_datasets = training_util.datasets_from_params(params)

    train_data = all_datasets['train']
    validation_data = all_datasets.get('validation')
    # test_data = all_datasets.get('test')

    train_candidates = []
    train_answers = []
    for instance in train_data:
        candidate = instance.fields['candidates'].array.tolist()
        train_candidates += candidate
        answer = candidate[
            instance.fields['answer'].label
        ]
        train_answers += [answer]

    with open("analysis.train.candidates.json", "w") as fh:
        json.dump(Counter(train_candidates), fh, indent=2)
    with open("analysis.train.answers.json", "w") as fh:
        json.dump(Counter(train_answers), fh, indent=2)

    valid_candidates = []
    valid_answers = []
    for instance in validation_data:
        candidate = instance.fields['candidates'].array.tolist()
        valid_candidates += candidate
        answer = candidate[
            instance.fields['answer'].label
        ]
        valid_answers += [answer]

    with open("analysis.valid.candidates.json", "w") as fh:
        json.dump(Counter(valid_candidates), fh, indent=2)
    with open("analysis.valid.answers.json", "w") as fh:
        json.dump(Counter(valid_answers), fh, indent=2)


if __name__ == '__main__':
    args = parse_args()
    main(args.param_path, args.overrides)
