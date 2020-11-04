import logging
from typing import Dict

from allennlp import modules
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.training.metrics import CategoricalAccuracy
import torch
from torch import nn
import torch.nn.functional as F

from src.module.seq2vec_encoders import PretrainedPooler

logger = logging.getLogger(__name__)


@Model.register('definition')
class Definition(Model):
    def __init__(self,
                 idiom_vector_path: str,
                 dropout: float,
                 vocab: Vocabulary,
                 content_embedder: TextFieldEmbedder,
                 option_vector_encoder: Seq2VecEncoder,
                 use_pretrained: bool = False,
                 use_idiom_embedding: bool = True,
                 use_idiom_text: bool = False,
                 use_idiom_definition: bool = False,
                 use_reasoner: bool = False,
                 idiom_vector_size: int = None) -> None:
        super().__init__(vocab)
        if idiom_vector_size is not None and use_pretrained:
            raise ValueError("When `use_pretrained` is True, `idiom_vector_size` must be None.")
        if not use_idiom_embedding and use_pretrained:
            raise ValueError("use_pretrained=True but use_idiom_embedding=False.")

        # suppose to be BERT model
        self.content_embedder = content_embedder
        self.option_vector_encoder = option_vector_encoder

        self.use_idiom_embedding = use_idiom_embedding

        if self.use_idiom_embedding:
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

        self.option_vector_encoder = option_vector_encoder

        if self.use_idiom_embedding:
            idiom_merger_in_features = self.option_embedder.get_output_dim()
        else:
            idiom_merger_in_features = 0

        self.use_idiom_text = use_idiom_text
        if self.use_idiom_text:
            idiom_merger_in_features += self.option_vector_encoder.get_output_dim()

        self.use_idiom_definition = use_idiom_definition
        if self.use_idiom_definition:
            idiom_merger_in_features += self.option_vector_encoder.get_output_dim()

        self.option_merger = nn.Linear(
            in_features=idiom_merger_in_features,
            out_features=self.content_embedder.get_output_dim(),
            bias=True
        )

        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Linear(self.content_embedder.get_output_dim(), 1)

        self.use_reasoner = use_reasoner
        if use_reasoner:
            embedding_size = self.content_embedder.get_output_dim()
            self.option_reasoner = StackedSelfAttentionEncoder(
                input_dim=embedding_size,
                hidden_dim=embedding_size,
                projection_dim=embedding_size,
                feedforward_hidden_dim=embedding_size,
                num_layers=1,
                num_attention_heads=2,
                use_positional_encoding=False
            )

        self.loss = nn.CrossEntropyLoss()
        self.acc = CategoricalAccuracy()

    def forward(self,
                content,
                blank_indices,
                candidates,
                candidate_texts,
                candidate_definitions,
                answer=None,
                meta=None) -> Dict[str, torch.Tensor]:
        # (batch_size, seq_len, embedding_size) :-> (B, S, E)
        embedded_content = self.content_embedder(content)
        if self.use_idiom_embedding:
            # (batch_size, 7, embedding_size) :-> (B, C, E)
            embedded_candidate = self.option_embedder(candidates)
            option_merger_in_features = [embedded_candidate]
        else:
            option_merger_in_features = []

        for enable, option_item in zip([self.use_idiom_text, self.use_idiom_definition],
                                       [candidate_texts, candidate_definitions]):
            if enable:
                if isinstance(self.option_vector_encoder, PretrainedPooler):
                    batch_size, option_size, seq_len = option_item['bert'].shape
                    in_option_item = option_item['bert'].view(-1, seq_len)
                    encoded_option = self.option_vector_encoder(
                        in_option_item,
                        (in_option_item != 0).float()
                    )
                    encoded_option = encoded_option.view(batch_size, option_size, -1)
                else:
                    berted_option = self.content_embedder(option_item)
                    batch_size, _, seq_len, embedding_size = berted_option.shape
                    encoded_option = self.option_vector_encoder(
                        berted_option.view(-1, seq_len, embedding_size),
                        option_item['mask'].view(-1, seq_len)
                    )
                    encoded_option = encoded_option.view(batch_size, -1, embedding_size)
                option_merger_in_features += [encoded_option]

        merged_candidate = torch.cat(option_merger_in_features, -1)
        encoded_candidate = self.option_merger(merged_candidate)

        # (batch_size, embedding_size) :-> (B, E)
        embedded_blanks = embedded_content[torch.arange(embedded_content.size(0),
                                                        dtype=torch.long,
                                                        device=embedded_content.device),
                                           blank_indices.squeeze(1)]
        # (batch_size, 7, embedding_size)
        hadamard_product = self.dropout(torch.einsum('bce,be->bce', encoded_candidate, embedded_blanks))
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
