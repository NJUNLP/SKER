import json
import re
from typing import Dict
from typing import Iterable
from typing import List

from allennlp.data import DatasetReader
from allennlp.data import Instance
from allennlp.data import Token
from allennlp.data.fields import ArrayField
from allennlp.data.fields import IndexField
from allennlp.data.fields import LabelField
from allennlp.data.fields import TextField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.token_indexers import TokenIndexer
import numpy as np
from overrides import overrides
from pytorch_pretrained_bert import BertTokenizer


@DatasetReader.register("baseline_reader")
class BaselineReader(DatasetReader):
    def __init__(self,
                 idiom_vector_path: str,
                 pretrained_model: str,
                 content_token_indexer: Dict[str, TokenIndexer],
                 max_seq_length: int = 512,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=False)
        self.content_token_indexer = content_token_indexer
        self.max_seq_length = max_seq_length

        idiom_list = []
        with open(idiom_vector_path) as fh:
            for line in fh:
                idiom_list.append(
                    line.strip().split()[0]
                )
        self.idiom_list = idiom_list

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as fh:
            for line in fh:
                example = json.loads(line)

                content = example["content"]
                candidates = example["candidates"]
                # 留空个数
                real_count = example.get("realCount", 1)
                # 真实答案
                ground_truths = example.get("groundTruth")

                yield from self.text_to_instance(
                    content,
                    candidates,
                    ground_truths=ground_truths,
                    real_count=real_count
                )

    @overrides
    def text_to_instance(self,
                         content: str,
                         candidates: List[str],
                         ground_truths: List[str] = None,
                         real_count: int = 1) -> Iterable[Instance]:
        splits = re.split(r'#idiom#', content)
        assert real_count + 1 == len(splits)
        assert real_count == len(candidates)
        split_tokens = [self.tokenizer.tokenize(item) for item in splits]
        for index, current_candidates in enumerate(candidates):
            before_part_tokens = [Token(token) for token in split_tokens[0]]
            for before_part in split_tokens[1:index+1]:
                before_part_tokens += [Token('[UNK]')] + [Token(token) for token in before_part]
            after_part_tokens = [Token(token) for token in split_tokens[index+1]]
            for after_part in split_tokens[index+2:]:
                after_part_tokens += [Token('[UNK]')] + [Token(token) for token in after_part]

            # 将 留空处 打上 [MASK]标记
            content_tokens = before_part_tokens + [Token('[MASK]')] + after_part_tokens

            # 取 留空 前后最多max_seq_length的内容作为输入
            half_length = self.max_seq_length // 2
            if len(before_part_tokens) < half_length:
                start = 0
                end = min(len(before_part_tokens) + 1 + len(after_part_tokens), self.max_seq_length - 2)
            elif len(after_part_tokens) < half_length:
                end = len(before_part_tokens) + 1 + len(after_part_tokens)
                start = max(0, end - (self.max_seq_length - 2))
            else:
                start = len(before_part_tokens) + 3 - half_length
                end = len(before_part_tokens) + 1 + half_length

            content_tokens = content_tokens[start:end]

            # 填空内容
            content_field = TextField(content_tokens, self.content_token_indexer)

            # 留空 的位置
            blank_index = content_tokens.index(Token("[MASK]"))
            blank_index_field = IndexField(blank_index, content_field)

            # 候选成语
            candidate_tokens = [
                self.idiom_list.index(option) for option in current_candidates
            ]
            candidate_tokens = np.array(candidate_tokens)
            candidate_field = ArrayField(candidate_tokens, dtype=np.long)

            fields = {
                "content": content_field,
                "blank_indices": blank_index_field,
                "candidates": candidate_field,
            }

            if ground_truths:
                label = current_candidates.index(ground_truths[index])
                label_field = LabelField(label, skip_indexing=True)
                fields["answer"] = label_field

                # 元信息
                meta = {
                    "content": '[UNK]'.join(splits[:index+1]) + "[MASK]" + '[UNK]'.join(splits[index+1:]),
                    "candidates": current_candidates,
                    "answer": ground_truths[index]
                }
            else:
                meta = {
                    "content": '[UNK]'.join(splits[:index+1]) + "[MASK]" + '[UNK]'.join(splits[index+1:]),
                    "candidates": current_candidates,
                }
            fields["meta"] = MetadataField(meta)

            yield Instance(fields)
