import json

import click


@click.command()
@click.argument('idioms', type=click.File('r'))
@click.argument('negs', type=click.File('r'))
@click.argument('graph', type=click.File('r'))
@click.argument('output', type=click.File('w'))
def print_nn_scd(idioms, negs, graph, output):
    idiom_scd = json.load(idioms)
    neg_scd = json.load(negs)
    graph = json.load(graph)

    result = {}

    for idiom in idiom_scd:
        result[idiom] = {
            'scd': idiom_scd[idiom],
            'negs': {
                item: neg_scd[item]
                for item in graph[idiom]['negs']
                if item in neg_scd
            }
        }

    json.dump(result, output, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    print_nn_scd()
