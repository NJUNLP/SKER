from allennlp.modules import FeedForward
from allennlp.modules import LayerNorm
from allennlp.modules.seq2seq_encoders import MultiHeadSelfAttention
from allennlp.nn import Activation
from overrides import overrides
import torch
from torch.nn import Dropout
from typing import List

from allennlp.modules import Seq2SeqEncoder


@Seq2SeqEncoder.register("gated_self_attention")
class GatedSelfAttention(Seq2SeqEncoder):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2,
                 attention_dropout_prob: float = 0.1) -> None:
        super().__init__()

        self._attention_layers: List[MultiHeadSelfAttention] = []
        self._feedfoward_layers: List[FeedForward] = []
        self._layer_norm_layers: List[LayerNorm] = []
        self._feed_forward_layer_norm_layers: List[LayerNorm] = []
        self._reset_gate_layers: List[FeedForward] = []

        feedfoward_input_dim = input_dim
        for i in range(num_layers):
            feedfoward = FeedForward(feedfoward_input_dim,
                                     activations=[Activation.by_name('relu')(),
                                                  Activation.by_name('linear')()],
                                     hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                     num_layers=2,
                                     dropout=dropout_prob)

            # Note: Please use `ModuleList` in new code. It provides better
            # support for running on multiple GPUs. We've kept `add_module` here
            # solely for backwards compatibility with existing serialized models.
            self.add_module(f"feedforward_{i}", feedfoward)
            self._feedfoward_layers.append(feedfoward)

            feedforward_layer_norm = LayerNorm(feedfoward.get_output_dim())
            self.add_module(f"feedforward_layer_norm_{i}", feedforward_layer_norm)
            self._feed_forward_layer_norm_layers.append(feedforward_layer_norm)

            self_attention = MultiHeadSelfAttention(num_heads=num_attention_heads,
                                                    input_dim=hidden_dim,
                                                    attention_dim=projection_dim,
                                                    values_dim=projection_dim,
                                                    attention_dropout_prob=attention_dropout_prob)
            self.add_module(f"self_attention_{i}", self_attention)
            self._attention_layers.append(self_attention)

            reset_gate = FeedForward(feedforward_hidden_dim,
                                     activations=Activation.by_name('sigmoid')(),
                                     hidden_dims=hidden_dim,
                                     num_layers=1,
                                     dropout=dropout_prob)
            self.add_module(f"reset_gate_{i}", reset_gate)
            self._reset_gate_layers.append(reset_gate)

            layer_norm = LayerNorm(self_attention.get_output_dim())
            self.add_module(f"layer_norm_{i}", layer_norm)
            self._layer_norm_layers.append(layer_norm)

            feedfoward_input_dim = hidden_dim

        self.dropout = Dropout(residual_dropout_prob)
        self._input_dim = input_dim
        self._output_dim = self._attention_layers[-1].get_output_dim()

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor): # pylint: disable=arguments-differ
        output = inputs
        for i in range(len(self._attention_layers)):
            # It's necessary to use `getattr` here because the elements stored
            # in the lists are not replicated by torch.nn.parallel.replicate
            # when running on multiple GPUs. Please use `ModuleList` in new
            # code. It handles this issue transparently. We've kept `add_module`
            # (in conjunction with `getattr`) solely for backwards compatibility
            # with existing serialized models.
            attention = getattr(self, f"self_attention_{i}")
            feedforward = getattr(self, f"feedforward_{i}")
            feedforward_layer_norm = getattr(self, f"feedforward_layer_norm_{i}")
            reset_gate = getattr(self, f"reset_gate_{i}")
            layer_norm = getattr(self, f"layer_norm_{i}")
            cached_input = output
            # Project output of attention encoder through a feedforward
            # network and back to the input size for the next layer.
            # shape (batch_size, timesteps, input_size)
            feedforward_output = feedforward(output)
            feedforward_output = self.dropout(feedforward_output)
            if feedforward_output.size() == cached_input.size():
                # First layer might have the wrong size for highway
                # layers, so we exclude it here.
                feedforward_output = feedforward_layer_norm(feedforward_output + cached_input)
            reset_gate_output = reset_gate(feedforward_output)
            # shape (batch_size, sequence_length, hidden_dim)
            attention_output = attention(feedforward_output, mask)
            output = layer_norm(self.dropout(reset_gate_output * attention_output) +
                                (1-reset_gate_output) * feedforward_output)

        return output

