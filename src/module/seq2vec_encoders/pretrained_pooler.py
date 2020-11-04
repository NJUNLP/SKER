from typing import Union

from allennlp.modules import Seq2VecEncoder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from overrides import overrides
from pytorch_pretrained_bert import BertModel
import torch
from torch import nn


@Seq2VecEncoder.register("pretrained_pooler")
class PretrainedPooler(Seq2VecEncoder):
    def __init__(self,
                 pretrained_model: Union[str, BertModel],
                 dropout: float = 0.0) -> None:
        super().__init__()

        if isinstance(pretrained_model, str):
            model = PretrainedBertModel.load(pretrained_model)
        else:
            model = pretrained_model

        self._dropout = torch.nn.Dropout(p=dropout)

        self.bert = model
        self._embedding_dim = model.config.hidden_size

        self.dense = nn.Linear(4 * self._embedding_dim, self._embedding_dim)
        self.activation = nn.Tanh()

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        encoded_layers, _ = self.bert(tokens, attention_mask=mask)
        sequence_output = torch.cat(
            [hidden_states[:, 0] for hidden_states in encoded_layers[-4:]], -1
        )
        pooled = self.dense(sequence_output)
        pooled = self.activation(pooled)
        pooled = self._dropout(pooled)
        return pooled
