import heapq
import json

import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        "Analyze Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "idiom_vector_path",
        help="path of the idiom vector file"
    )
    parser.add_argument(
        "idiom_graph_path",
        help="path of the idiom graph file"
    )
    parser.add_argument(
        "--threshold", default=0.65, type=float
    )
    parser.add_argument(
        "--max_num", default=7, type=int
    )
    return parser.parse_args()


def main(idiom_vector_path, idiom_graph_path,
         threshold=0.65, max_num=200):
    idiom_vectors = []
    with open(idiom_vector_path) as fh:
        for line in fh:
            idiom_vectors.append(
                list(map(float, line.strip().split()[1:]))
            )
    idiom_vectors = np.array(idiom_vectors)
    distance = idiom_vectors @ idiom_vectors.T
    norms = np.linalg.norm(idiom_vectors, axis=1)
    cosine = distance / (norms[:, np.newaxis] * norms[np.newaxis, :])
    cosine = cosine - np.eye(cosine.shape[0])

    indices = []
    for line in cosine:
        neighbor_indices = heapq.nlargest(max_num, range(len(line)), line.take)
        neighbor_indices = [
            index for index in neighbor_indices
            if line[index] >= threshold
        ]
        indices.append(neighbor_indices)

    with open(idiom_graph_path, "w") as fh:
        json.dump(indices, fh, indent=2)


if __name__ == '__main__':
    args = parse_args()
    main(args.idiom_vector_path, args.idiom_graph_path,
         threshold=args.threshold, max_num=args.max_num)
