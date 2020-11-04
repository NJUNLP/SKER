from allennlp import modules
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.nn import Activation
from allennlp.training.metrics import CategoricalAccuracy
import logging
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict
from typing import List

from src.module.graph_embdder import GraphEmbedder
from src.module.seq2vec_encoders.first_vec_encoder import FirstVecEncoder

logger = logging.getLogger(__name__)


@Model.register('full_ran')
class Full(Model):
    def __init__(self,
                 idiom_vector_path: str,
                 idiom_graph_path: str,
                 dropout: float,
                 vocab: Vocabulary,
                 content_embedder: TextFieldEmbedder,
                 neighbor_num: int = 7,
                 mode: List[str] = None) -> None:
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

        self.graph_embedder = GraphEmbedder(idiom_graph_path,
                                            neighbor_num=neighbor_num,
                                            drop_neighbor=False)

        embedding_dim = self.content_embedder.get_output_dim()
        self.option_embedder = modules.Embedding(
            num_embeddings=len(idiom_list),
            embedding_dim=embedding_dim,
            # 使用 预训练的成语向量
            # weight=torch.FloatTensor(idiom_vectors)
        )

        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Linear(self.content_embedder.get_output_dim(), 1)

        embedding_size = self.content_embedder.get_output_dim()

        self.neighbour_reasoner = StackedSelfAttentionEncoder(
            input_dim=embedding_size,
            hidden_dim=embedding_size,
            projection_dim=embedding_size,
            feedforward_hidden_dim=embedding_size,
            num_layers=1,
            num_attention_heads=2,
            use_positional_encoding=False
        )
        self.option_encoder = FirstVecEncoder(
            embedding_dim=embedding_size
        )

        self.option_reasoner = StackedSelfAttentionEncoder(
            input_dim=embedding_size,
            hidden_dim=embedding_size,
            projection_dim=embedding_size,
            feedforward_hidden_dim=embedding_size,
            num_layers=1,
            num_attention_heads=2,
            use_positional_encoding=False
        )

        if mode is None:
            mode = ['raw', 'ocn', 'nn']
        else:
            for item in mode:
                assert item in ['raw', 'ocn', 'nn'], f"{item} is invalid"
        self.mode = mode

        self.data_merger = FeedForward(
            input_dim=embedding_size*len(mode),
            num_layers=1,
            hidden_dims=embedding_size,
            activations=Activation.by_name('linear')(),
            dropout=0.1
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
        # (batch_size, candidate_num, embedding_size) :-> (B, C, E)
        embedded_candidate = self.option_embedder(candidates)
        # (batch_size, embedding_size) :-> (B, E)
        embedded_blanks = embedded_content[torch.arange(embedded_content.size(0),
                                                        dtype=torch.long,
                                                        device=embedded_content.device),
                                           blank_indices.squeeze(1)]

        ###########
        # 自身节点 #
        ###########
        original_results = self.dropout(torch.einsum('bce,be->bce', embedded_candidate, embedded_blanks))

        ###########
        # 邻居节点 #
        ###########
        # (batch_size, candidate_num, 1+candidate_neighbor_num) :-> (B, C, N+1)
        candidate_neighbors, candidate_neighbor_masks = self.graph_embedder(candidates,
                                                                            self.training)
        candidate_neighbors = candidate_neighbors.long()
        candidate_neighbor_masks = candidate_neighbor_masks.long()
        # (batch_size, candidate_num, 1+candidate_neighbor_num, embedding_size) :-> (B, C, N+1, E)
        embedded_candidate_neighbors = self.option_embedder(candidate_neighbors)

        # (batch_size, candidate_num, 1+candidate_neighbor_num, embedding_size) :-> (B, C, N+1, E)
        neighbor_results = self.dropout(torch.einsum('bcne,be->bcne',
                                                     embedded_candidate_neighbors, embedded_blanks))
        batch_size, candidate_num, candidate_neighbor_num, embedding_size = neighbor_results.shape
        # (batch_size * candidate_num, 1+candidate_neighbor_num, embedding_size) :-> (B*C, N+1, E)
        neighbor_results = neighbor_results.view(-1, candidate_neighbor_num, embedding_size)
        # (B*C, N+1)
        candidate_neighbor_masks = candidate_neighbor_masks.view(-1, candidate_neighbor_num)
        # (B*C, N+1, E)
        attentive_neighbor = self.neighbour_reasoner(
            neighbor_results, mask=candidate_neighbor_masks
        )
        # (B*C, E)
        attentive_neighbor = self.option_encoder(
            attentive_neighbor, mask=candidate_neighbor_masks
        )
        attentive_neighbor = attentive_neighbor.view(batch_size, candidate_num, -1)

        ###########
        # 选项节点 #
        ###########
        attentive_candidate = self.option_reasoner(
            original_results, mask=None
        )

        final_candidates = self.data_merger(
            torch.cat(
                [embedded_candidate, attentive_candidate, attentive_neighbor],
                dim=2
            )
        )

        logits = self.scorer(final_candidates).squeeze(2)

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
