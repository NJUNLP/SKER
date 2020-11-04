from allennlp import modules
from allennlp.data import Vocabulary
from allennlp.models import load_archive
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
from typing import Optional

from src.model import Baseline
from src.module.graph_embdder import GraphEmbedder
from src.module.seq2vec_encoders import GatedSelfAttention
from src.module.seq2vec_encoders.first_vec_encoder import FirstVecEncoder

logger = logging.getLogger(__name__)


@Model.register('gan')
class GAN(Model):
    def __init__(self,
                 idiom_vector_path: str,
                 idiom_graph_path: str,
                 dropout: float,
                 vocab: Vocabulary,
                 content_embedder: TextFieldEmbedder,
                 use_pretrained: bool = False,
                 idiom_vector_size: int = None,
                 neighbor_num: int = 7,
                 drop_neighbor: bool = False,
                 reasoner_mode: str = None,
                 use_reasoner: bool = False,
                 teacher_model_path: str = None,
                 data_mode: str = None,
                 num_neighbour_attention_heads: int = 2) -> None:
        super().__init__(vocab)
        self.content_embedder = content_embedder

        if idiom_vector_size is not None and use_pretrained:
            raise ValueError("When `use_pretrained` is True, `idiom_vector_size` must be None.")

        self.teacher_model_path = teacher_model_path
        self.teacher = self.load_teacher()

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
                                            drop_neighbor=drop_neighbor)

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
                num_attention_heads=num_neighbour_attention_heads,
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

        self.option_encoder = FirstVecEncoder(
            embedding_dim=embedding_size
        )

        self.use_reasoner = use_reasoner
        if use_reasoner:
            if reasoner_mode == "self_attention":
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

        if data_mode is None:
            data_mode = 'pipeline'
        else:
            data_mode = data_mode.lower()
        assert data_mode in ('pipeline', 'parallel')
        self.data_mode = data_mode
        if self.use_reasoner and self.data_mode == 'parallel':
            self.data_merger = FeedForward(
                input_dim=embedding_size+embedding_size,
                num_layers=1,
                hidden_dims=embedding_size,
                activations=Activation.by_name('linear')(),
                dropout=0.1
            )

        if self.teacher is not None:
            self.option_embedder.weight = self.teacher.option_embedder.weight
            self.option_embedder.weight.requires_grad = True

            self.scorer.weight = self.teacher.scorer.weight
            self.scorer.weight.requires_grad = True

        self.loss = nn.CrossEntropyLoss()
        self.acc = CategoricalAccuracy()

    def forward(self,
                content,
                blank_indices,
                candidates,
                answer=None,
                meta=None) -> Dict[str, torch.Tensor]:
        # Step 1: Encoding Passage
        # (batch_size, seq_len, embedding_size) :-> (B, S, E)
        embedded_content = self.content_embedder(content)
        # (batch_size, candidate_num, embedding_size) :-> (B, C, E)
        embedded_candidate = self.option_embedder(candidates)
        # Step 2: Retrieve Neighbors of Candidates
        # (batch_size, candidate_num, 1+candidate_neighbor_num) :-> (B, C, N+1)
        candidate_neighbors, candidate_neighbor_masks, meta_graph = self.graph_embedder(
            candidates, self.training, verbose=not self.training)
        candidate_neighbors = candidate_neighbors.long()
        candidate_neighbor_masks = candidate_neighbor_masks.long()
        # logger.info(f'candidate_neighbors.shape={candidate_neighbors.shape}')
        # logger.info(f'candidate_neighbors:\n{candidate_neighbors}')
        # logger.info(f'candidate_neighbor_masks:\n{candidate_neighbor_masks}')
        # Step 3: Embedding Candidates' Neighbors (containing candidates)
        # (batch_size, candidate_num, 1+candidate_neighbor_num, embedding_size) :-> (B, C, N+1, E)
        embedded_candidate_neighbors = self.option_embedder(candidate_neighbors)

        # Step 4: Retrieve the Blank of passages
        # (batch_size, embedding_size) :-> (B, E)
        embedded_blanks = embedded_content[torch.arange(embedded_content.size(0),
                                                        dtype=torch.long,
                                                        device=embedded_content.device),
                                           blank_indices.squeeze(1)]

        # Step 5: Interaction between Candidates and Blanks
        # (batch_size, candidate_num, 1+candidate_neighbor_num, embedding_size) :-> (B, C, N+1, E)
        hadamard_product = self.dropout(torch.einsum('bcne,be->bcne',
                                                     embedded_candidate_neighbors, embedded_blanks))
        batch_size, candidate_num, candidate_neighbor_num, embedding_size = hadamard_product.shape
        # (batch_size * candidate_num, 1+candidate_neighbor_num, embedding_size) :-> (B*C, N+1, E)
        hadamard_product = hadamard_product.view(-1, candidate_neighbor_num, embedding_size)
        # (B*C, N+1)
        candidate_neighbor_masks = candidate_neighbor_masks.view(-1, candidate_neighbor_num)
        if self.teacher is not None:
            # (B*C, N+1)
            before_logits = self.scorer(hadamard_product).squeeze(2)
            neighbor_scores = torch.sigmoid(before_logits)
            hadamard_product = neighbor_scores.unsqueeze(-1) * hadamard_product
        # Step 6: GAT (interaction between candidates and their corresponding neighbors)
        # (B*C, N+1, E)
        # 可视化
        # attentive_neighbor, visual_attention_neighbor = self.option_reasoner(
        #     hadamard_product, mask=candidate_neighbor_masks
        # )
        attentive_neighbor = self.option_reasoner(
                hadamard_product, mask=candidate_neighbor_masks
        )
        # logger.info(f'visual_attention_neighbor=\n{visual_attention_neighbor}\n')
        # logger.info(f'candidate_neighbor_masks=\n{candidate_neighbor_masks}\n')
        # (B*C, E)
        attentive_neighbor = self.option_encoder(
            attentive_neighbor, mask=candidate_neighbor_masks
        )
        attentive_neighbor = attentive_neighbor.view(batch_size, candidate_num, -1)
        if self.use_reasoner:
            if self.data_mode == 'parallel':
                attentive_candidate = embedded_candidate
            else:
                attentive_candidate = attentive_neighbor
            if self.teacher is not None:
                before_candidate_logits = self.scorer(attentive_candidate).squeeze(2)
                candidate_scores = torch.sigmoid(before_candidate_logits)
                attentive_candidate = candidate_scores.unsqueeze(-1) * attentive_candidate
            attentive_candidate = self.option_reasoner(
                attentive_candidate, mask=None
            )
            if self.data_mode == 'parallel':
                attentive_candidate = self.data_merger(
                    torch.cat([attentive_candidate, attentive_neighbor], dim=2)
                )
            logits = self.scorer(attentive_candidate).squeeze(2)
        else:
            # Step 7: make a prediction
            # (batch_size, C)
            logits = self.scorer(attentive_neighbor).squeeze(2)

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
            output["graph"] = meta_graph
            # 可视化
            # _, attention_head, candidate_neighbor_num = visual_attention_neighbor.shape
            # output["visual_attention_graph"] = visual_attention_neighbor.view(batch_size, -1, attention_head, candidate_neighbor_num)
            output["mask"] = candidate_neighbor_masks.view(batch_size, -1, candidate_neighbor_num)

        if answer is not None:
            loss = self.loss(logits, answer.long())
            output["loss"] = loss
            self.acc(logits, answer)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "accuracy": self.acc.get_metric(reset=reset)
        }

    def load_teacher(self) -> Optional[Baseline]:
        if self.teacher_model_path is None:
            return None
        else:
            archive = load_archive(self.teacher_model_path)
            model = archive.model
            if getattr(model, 'use_reasoner', None) is not None:
                assert not model.use_reasoner, "teacher should not use_reasoner"
            return model
