from allennlp import modules
from typing import Dict

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
import torch
from torch import nn

import torch.nn.functional as F


@Model.register('pretrain')
class Pretrain(Model):
    def __init__(self,
                 idiom_vector_path: str,
                 dropout: float,
                 vocab: Vocabulary,
                 content_embedder: TextFieldEmbedder) -> None:
        super().__init__(vocab)
        self.content_embedder = content_embedder

        idiom_list, idiom_vectors = [], []
        with open(idiom_vector_path) as fh:
            for line in fh:
                idiom_list.append(
                    line.strip().split()[0]
                )
                idiom_vectors.append(
                    list(map(float, line.strip().split()[1:]))
                )

        self.option_embedder = modules.Embedding(
            num_embeddings=len(idiom_list),
            embedding_dim=len(idiom_vectors[0]),
            projection_dim=self.content_embedder.get_output_dim(),
            # 使用 预训练的成语向量
            weight=torch.tensor(idiom_vectors, dtype=torch.float, requires_grad=True)
        )

        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Linear(self.content_embedder.get_output_dim(), 1)

        self.loss = nn.CrossEntropyLoss()
        self.acc = CategoricalAccuracy()

    def forward(self,
                content,
                blank_indices,
                candidates,
                answer=None,
                meta=None) -> Dict[str, torch.Tensor]:
        # (batch_size, seq_len, embedding_size) :-> (B, S, E)
        embedded_content = self.content_embedder(content)
        # (batch_size, 7, embedding_size) :-> (B, C, E)
        embedded_candidate = self.option_embedder(candidates)
        # (batch_size, embedding_size) :-> (B, E)
        embedded_blanks = embedded_content[torch.arange(embedded_content.size(0),
                                                        dtype=torch.long,
                                                        device=embedded_content.device),
                                           blank_indices.squeeze(1)]
        # (batch_size, 7, embedding_size)
        hadamard_product = self.dropout(torch.einsum('bce,be->bce', embedded_candidate, embedded_blanks))
        # (batch_size, 7)
        logits = self.scorer(hadamard_product).squeeze(2)
        # (batch_size, )
        probs = F.softmax(logits, dim=-1)

        preds = torch.argmax(probs, dim=-1).cpu().detach().numpy().tolist()

        output = {
            "probs": probs.cpu().detach().numpy().tolist(),
            "preds": preds
        }

        if meta is not None:
            for item, pred in zip(meta, preds):
                item['prediction'] = item['candidates'][pred]
            output["meta"] = meta

        if answer is not None:
            loss = self.loss(logits, answer.long())
            output["loss"] = loss
            self.acc(logits, answer)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "accuracy": self.acc.get_metric(reset=reset)
        }
