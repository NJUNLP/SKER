import json

import click
import numpy as np


@click.command()
@click.argument('scd', type=click.File('r'))
def analyze_scd(scd):
    scd = json.load(scd)

    # idiom总数
    print(f'idiom总数: {len(scd)}')
    lst = [
        len(item['negs'])
        for item in scd.values()
    ]
    # 近邻个数平均数
    print(f'近邻个数平均数: {np.mean(lst)}')
    # 近邻个数中位数
    print(f'近邻个数中位数: {np.median(lst)}')
    # 近邻个数众数
    # counts = np.bincount(lst)
    # print(f'近邻个数众数: {np.argmax(counts)}')
    # 近邻个数为0的个数及比重
    count = 0
    for item in scd.values():
        if len(item['negs']) == 0:
            count += 1
    print(f'近邻个数为0的个数及比重: {count}/{len(scd)}={count/len(scd)}')

    # 在排除近邻个数为0的成语情况下
    print('在排除近邻个数为0的成语情况下')

    # 在不考虑自身的情况下
    print('在不考虑自身的情况下')
    # 近邻的SCD均值相较于原成语上升的个数及比重
    count = 0
    total = 0
    for item in scd.values():
        if len(item['negs']) == 0:
            continue
        total += 1

        origin = np.mean(item['scd'])
        nn_scd = 0
        for nn in item['negs']:
            nn_scd += np.mean(item['negs'][nn])
        nn_scd /= len(item['negs'])
        if nn_scd > origin:
            count += 1
    print(f'近邻的SCD均值相较于原成语上升的个数及比重: {count}/{total}={count / total}')
    # 近邻的SCD均值相较于原成语上升（含等于）的个数及比重
    count = 0
    total = 0
    for item in scd.values():
        if len(item['negs']) == 0:
            continue
        total += 1

        origin = np.mean(item['scd'])
        nn_scd = 0
        for nn in item['negs']:
            nn_scd += np.mean(item['negs'][nn])
        nn_scd /= len(item['negs'])
        if nn_scd >= origin:
            count += 1
    print(f'近邻的SCD均值相较于原成语上升（含等于）的个数及比重: {count}/{total}={count / total}')
    # 近邻的SCD最大值相较于原成语上升的个数及比重
    count = 0
    total = 0
    for item in scd.values():
        if len(item['negs']) == 0:
            continue
        total += 1

        origin = np.mean(item['scd'])
        nn_scd = 0
        for nn in item['negs']:
            nn_scd = max(nn_scd, np.mean(item['negs'][nn]))
        if nn_scd > origin:
            count += 1
    print(f'近邻的SCD最大值相较于原成语上升的个数及比重: {count}/{total}={count / total}')
    # 近邻的SCD最大值相较于原成语上升（含等于）的个数及比重
    count = 0
    total = 0
    for item in scd.values():
        if len(item['negs']) == 0:
            continue
        total += 1

        origin = np.mean(item['scd'])
        nn_scd = 0
        for nn in item['negs']:
            nn_scd = max(nn_scd, np.mean(item['negs'][nn]))
        if nn_scd >= origin:
            count += 1
    print(f'近邻的SCD最大值相较于原成语上升（含等于）的个数及比重: {count}/{total}={count / total}')

    # 在考虑自身的情况下
    print('在考虑自身的情况下')
    # 近邻的SCD均值相较于原成语上升的个数及比重
    count = 0
    total = 0
    for item in scd.values():
        if len(item['negs']) == 0:
            continue
        total += 1

        origin = np.mean(item['scd'])
        nn_scd = origin
        for nn in item['negs']:
            nn_scd += np.mean(item['negs'][nn])
        nn_scd /= (len(item['negs'])+1)
        if nn_scd > origin:
            count += 1
    print(f'近邻的SCD均值相较于原成语上升的个数及比重: {count}/{total}={count / total}')
    # 近邻的SCD均值相较于原成语上升（含等于）的个数及比重
    count = 0
    total = 0
    for item in scd.values():
        if len(item['negs']) == 0:
            continue
        total += 1

        origin = np.mean(item['scd'])
        nn_scd = origin
        for nn in item['negs']:
            nn_scd += np.mean(item['negs'][nn])
        nn_scd /= (len(item['negs'])+1)
        if nn_scd >= origin:
            count += 1
    print(f'近邻的SCD均值相较于原成语上升（含等于）的个数及比重: {count}/{total}={count / total}')
    # 近邻的SCD最大值相较于原成语上升的个数及比重
    count = 0
    total = 0
    for item in scd.values():
        if len(item['negs']) == 0:
            continue
        total += 1

        origin = np.mean(item['scd'])
        nn_scd = origin
        for nn in item['negs']:
            nn_scd = max(nn_scd, np.mean(item['negs'][nn]))
        if nn_scd > origin:
            count += 1
    print(f'近邻的SCD最大值相较于原成语上升的个数及比重: {count}/{total}={count / total}')
    # 近邻的SCD最大值相较于原成语上升（含等于）的个数及比重
    count = 0
    total = 0
    for item in scd.values():
        if len(item['negs']) == 0:
            continue
        total += 1

        origin = np.mean(item['scd'])
        nn_scd = origin
        for nn in item['negs']:
            nn_scd = max(nn_scd, np.mean(item['negs'][nn]))
        if nn_scd >= origin:
            count += 1
    print(f'近邻的SCD最大值相较于原成语上升（含等于）的个数及比重: {count}/{total}={count / total}')


if __name__ == '__main__':
    analyze_scd()
