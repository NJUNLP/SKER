import json

import click


@click.command()
@click.argument('baseline', type=click.STRING)
@click.argument('model', type=click.STRING)
# @click.argument('r2r', type=click.File('w'))
# @click.argument('r2w', type=click.File('w'))
@click.argument('w2r', type=click.File('w'))
# @click.argument('w2w', type=click.File('w'))
def case_study(baseline, model, w2r):
    cnt_r2r = 0
    cnt_r2w = 0
    cnt_w2r = 0
    cnt_w2w = 0

    with open(baseline) as baseline, open(model) as model:
        for b_line, m_line in zip(baseline, model):
            b_item = json.loads(b_line)
            m_item = json.loads(m_line)
            b_is_r = b_item['meta']['answer'] == b_item['meta']['prediction']
            m_is_r = m_item['meta']['answer'] == m_item['meta']['prediction']
            if b_is_r and m_is_r:
                cnt_r2r += 1
            if b_is_r and not m_is_r:
                cnt_r2w += 1
            if not b_is_r and m_is_r:
                cnt_w2r += 1
                w2r.write(f'{b_line.strip()}\n{m_line.strip()}\n')
            if not b_is_r and not m_is_r:
                cnt_w2w += 1
    print(
        f'cnt_r2r: {cnt_r2r}\n'
        f'cnt_r2w: {cnt_r2w}\n'
        f'cnt_w2r: {cnt_w2r}\n'
        f'cnt_w2w: {cnt_w2w}\n'
    )


if __name__ == '__main__':
    case_study()
