import json

import click


@click.command()
@click.argument('w2r', type=click.File('r'))
@click.argument('idx2idiom', type=click.File('r'))
@click.argument('transformed_w2r', type=click.File('w'))
def transform_idiom_in_w2r(
        w2r, idx2idiom, transformed_w2r
):
    idx2idiom = json.load(idx2idiom)
    for baseline in w2r:
        baseline = baseline.strip()
        model = json.loads(w2r.readline())
        graph = model['graph']
        model['graph'] = [
            [idx2idiom[idx] for idx in instance]
            for instance in graph
        ]
        model = json.dumps(model, ensure_ascii=False)
        transformed_w2r.write(f'{baseline}\n{model}\n')


if __name__ == '__main__':
    transform_idiom_in_w2r()
