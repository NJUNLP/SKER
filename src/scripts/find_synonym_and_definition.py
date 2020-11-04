import json

import click
from toolz import concat


@click.command()
@click.argument('indices', type=click.File('r'))
@click.argument('graph', type=click.File('r'))
@click.argument('definition', type=click.File('r'))
@click.argument('interest', type=click.File('r'))
@click.argument('output', type=click.File('w'))
def find_synonym_and_definition(
        indices,  # 近义索引
        graph,  # 近义词图
        definition,  # 成语及其释义
        interest,  # 感兴趣的成语
        output,   # 输出文件
):
    idiom2index = {}
    index2idiom = {}
    for index, line in enumerate(indices.readlines()):
        idiom = line.strip().split()[0]
        idiom2index[idiom] = index
        index2idiom[index] = idiom
    graph = json.load(graph)
    definition = json.load(definition)
    interest = [line.strip() for line in interest.readlines()]

    sampled = set(concat(
        [
            graph[idiom2index[idiom]]
            for idiom in interest
        ]
    ))

    sampled = set(
        index2idiom[index]
        for index in sampled
    ) - set(interest)

    result = {
        idiom: {
            "def": definition[idiom],
            "scd":0
        }
        for idiom in sampled
    }

    json.dump(result, output, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    find_synonym_and_definition()
