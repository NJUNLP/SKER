from allennlp import modules
from allennlp.data import Vocabulary
from allennlp.models import load_archive
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.training.metrics import CategoricalAccuracy
import logging
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict
from typing import Optional

from src.model import Baseline

logger = logging.getLogger(__name__)


@Model.register('denoise')
class Denoise(Model):
    def __init__(self,
                 idiom_vector_path: str,
                 dropout: float,
                 vocab: Vocabulary,
                 content_embedder: TextFieldEmbedder,
                 use_pretrained: bool = False,
                 use_reasoner: bool = False,
                 idiom_vector_size: int = None,
                 denoise_mode: str = 'soft',
                 denoise_lambda: float = None,
                 teacher_model_path: str = None,
                 teacher_mode: str = None) -> None:
        super().__init__(vocab)
        self.content_embedder = content_embedder

        if idiom_vector_size is not None and use_pretrained:
            raise ValueError("When `use_pretrained` is True, `idiom_vector_size` must be None.")

        if teacher_mode is not None:
            teacher_mode = teacher_mode.lower()
            assert teacher_mode in ('initialization', 'teacher'), (f'teacher_mode ({teacher_mode}) '
                                                                   'not in ("initialization", "teacher").')
        if teacher_mode is not None and teacher_model_path is None:
            raise ValueError("Please set teacher_model_path when teacher_mode is not None.")
        self.teacher_mode = teacher_mode
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

        if use_reasoner:
            logger.info(f"{type(self)} always uses the reasoner.")

        self.use_reasoner = True

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

        if self.teacher_mode == 'initialization':
            self.option_embedder.weight = self.teacher.option_embedder.weight
            self.option_embedder.weight.requires_grad = True

            self.scorer.weight = self.teacher.scorer.weight
            self.scorer.weight.requires_grad = True

        denoise_mode = denoise_mode.lower()
        assert denoise_mode in ('soft', 'hard', 'both', 'lambda'), (f'denoise_mode ({denoise_mode}) '
                                                                    'not in ("soft", "hard", "both", "lambda").')
        self.denoise_mode = denoise_mode
        self.denoise_lambda = denoise_lambda

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
        if self.teacher_mode == 'teacher':
            # (batch_size, 7)
            teacher_output = self.teacher.forward(content, blank_indices, candidates)
            candidate_scores = teacher_output["probs"]
            hadamard_product = candidate_scores.unsqueeze(-1) * hadamard_product
            # This class always use the reasoner.
            attentive_candidate = self.option_reasoner(
                hadamard_product, mask=None
            )
        else:
            # (batch_size, 7)
            before_logits = self.scorer(hadamard_product).squeeze(2)
            candidate_scores = torch.sigmoid(before_logits)
            if self.denoise_mode in ('hard', 'both'):
                attentive_candidate_mask = candidate_scores < 0.5
                if not attentive_candidate_mask.any():
                    attentive_candidate_mask = None
            else:
                attentive_candidate_mask = None
            if self.denoise_mode in ('soft', 'both'):
                hadamard_product = candidate_scores.unsqueeze(-1) * hadamard_product
            # This class always use the reasoner.
            attentive_candidate = self.option_reasoner(
                hadamard_product, mask=attentive_candidate_mask
            )
        # (batch_size, 7)
        logits = self.scorer(attentive_candidate).squeeze(2)
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
            if self.denoise_lambda is not None:
                before_loss = self.loss(before_logits, answer.long())
                loss += self.denoise_lambda * before_loss
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
