from allennlp import modules
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.training.metrics import CategoricalAccuracy
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict

from src.module.seq2vec_encoders import GatedSelfAttention


@Model.register('baseline')
class Baseline(Model):
    def __init__(self,
                 idiom_vector_path: str,
                 dropout: float,
                 vocab: Vocabulary,
                 content_embedder: TextFieldEmbedder,
                 use_pretrained: bool = False,
                 use_reasoner: bool = False,
                 idiom_vector_size: int = None,
                 reasoner_mode: str = None) -> None:
        super().__init__(vocab)
        self.content_embedder = content_embedder

        if idiom_vector_size is not None and use_pretrained:
            raise ValueError("When `use_pretrained` is True, `idiom_vector_size` must be None.")

        idiom_list, idiom_vectors = [], []
        with open(idiom_vector_path) as fh:
            for line in fh:
                idiom_list.append(
                    line.strip().split()[0]
                )
                idiom_vectors.append(
                    list(map(float, line.strip().split()[1:]))
                )

        self.use_pretrained = use_pretrained

        if self.use_pretrained:
            self.option_embedder = modules.Embedding(
                num_embeddings=len(idiom_list),
                embedding_dim=len(idiom_vectors[0]),
                projection_dim=self.content_embedder.get_output_dim(),
                # 使用 预训练的成语向量
                weight=torch.FloatTensor(idiom_vectors)
            )
        else:
            embedding_dim = idiom_vector_size or len(idiom_vectors[0])
            self.option_embedder = modules.Embedding(
                num_embeddings=len(idiom_list),
                embedding_dim=embedding_dim,
                projection_dim=self.content_embedder.get_output_dim(),
                # 使用 预训练的成语向量
                # weight=torch.FloatTensor(idiom_vectors)
            )

        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Linear(self.content_embedder.get_output_dim(), 1)

        self.use_reasoner = use_reasoner
        if use_reasoner:
            embedding_size = self.content_embedder.get_output_dim()
            if reasoner_mode is None:
                reasoner_mode = 'self_attention'
            else:
                reasoner_mode = reasoner_mode.lower()
                assert reasoner_mode in ('self_attention', 'gated_self_attention')
            self.reasoner_mode = reasoner_mode
            if reasoner_mode == 'self_attention':
                self.option_reasoner = StackedSelfAttentionEncoder(
                    input_dim=embedding_size,
                    hidden_dim=embedding_size,
                    projection_dim=embedding_size,
                    feedforward_hidden_dim=embedding_size,
                    num_layers=1,
                    num_attention_heads=2,
                    use_positional_encoding=False
                )
            elif reasoner_mode == "gated_self_attention":
                self.option_reasoner = GatedSelfAttention(
                    input_dim=embedding_size,
                    hidden_dim=embedding_size,
                    projection_dim=embedding_size,
                    feedforward_hidden_dim=embedding_size,
                    num_layers=1,
                    num_attention_heads=2
                )

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
        if self.use_reasoner:
            attentive_candidate = self.option_reasoner(
                hadamard_product, mask=None
            )
            # (batch_size, 7)
            logits = self.scorer(attentive_candidate).squeeze(2)
        else:
            # (batch_size, 7)
            logits = self.scorer(hadamard_product).squeeze(2)
        # (batch_size, )
        probs = F.softmax(logits, dim=-1)

        preds = torch.argmax(probs, dim=-1)

        output = {
            "probs": probs,
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
