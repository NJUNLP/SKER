from allennlp.modules import Seq2VecEncoder
import torch


@Seq2VecEncoder.register("first_vec")
class FirstVecEncoder(Seq2VecEncoder):
    def __init__(self,
                 embedding_dim: int) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        first_token_tensor = tokens[:, 0]
        return first_token_tensor
