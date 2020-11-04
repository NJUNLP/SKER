import json
import logging

import torch
from torch import nn
from torch.nn.functional import embedding

logger = logging.getLogger(__name__)


class GraphEmbedder(nn.Module):
    def __init__(self,
                 idiom_graph_path,
                 neighbor_num: int = 7,
                 include_self: bool = True,
                 drop_neighbor: bool = False):
        super().__init__()
        self.idiom_graph_path = idiom_graph_path
        with open(idiom_graph_path) as fh:
            idiom_graph = json.load(fh)
        if 'graph' in idiom_graph:
            idiom_graph = idiom_graph['graph']
        self.idiom_graph = idiom_graph
        self.neighbor_num = neighbor_num
        self.include_self = include_self
        self.drop_neighbor = drop_neighbor

        if include_self:
            neighbor_num += 1

        graph, graph_mask = [], []
        for index, line in enumerate(self.idiom_graph):
            if len(line) >= neighbor_num:
                mask = [1] * neighbor_num
            else:
                mask = [1] * len(line) + [0] * (neighbor_num - len(line))
            line = line[:neighbor_num]
            line += [0] * (neighbor_num - len(line))

            if include_self:
                mask = [1] + mask
                line = [index] + line
            graph_mask.append(mask)
            graph.append(line)
        graph = torch.tensor(graph, dtype=torch.float)
        graph_mask = torch.tensor(graph_mask, dtype=torch.float)
        self.graph = torch.nn.Parameter(graph, requires_grad=False)
        self.graph_mask = torch.nn.Parameter(graph_mask, requires_grad=False)

    def forward(self, candidates: torch.LongTensor, training: bool = False,
                exclude: bool = False, verbose: bool = False):
        # candidates: (batch_size, option_number) :-> (B, C)
        # neighbors: (batch_size, option_number, neighbor_num)
        neighbors = embedding(candidates, self.graph, norm_type=None)
        # mask: (batch_size, option_number, neighbor_num)
        masks = embedding(candidates, self.graph_mask, norm_type=None)
        if self.drop_neighbor and training:
            first = torch.zeros_like(masks).byte()
            first[:, :, 0] = 1
            masks = masks * (first | (torch.rand(masks.size()).to(masks.device) < 0.1)).float()
        if not verbose:
            return neighbors, masks, None
        else:
            batch_options = []
            for instance, instance_mask in zip(neighbors.long().cpu().numpy().tolist(),
                                               masks.long().cpu().numpy().tolist()):
                instance_options = []
                for option, option_mask in zip(instance, instance_mask):
                    idioms = [
                        idiom for idiom, idiom_mask in zip(option, option_mask)
                        if idiom_mask != 0
                    ]
                    instance_options.append(idioms)
                batch_options.append(instance_options)
            return neighbors, masks, batch_options
