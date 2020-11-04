import json

import click


@click.command()
@click.argument('scd', type=click.File('w'))
@click.argument('annotations', type=click.File('r'), nargs=-1)
def print_scd(scd, annotations):
    inner = []
    for annotation in annotations:
        inner.append(
            json.load(annotation)
        )

    first = inner[0]
    result = {}
    for idiom in first:
        result[idiom] = [
            item[idiom]['scd'] for item in inner
        ]

    json.dump(result, scd, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    print_scd()
