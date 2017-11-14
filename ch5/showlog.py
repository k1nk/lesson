import json
from matplotlib import pyplot


def load_log(path):
    with open(path) as f:
        # read json file
        ch_log = json.load(f)

    x = [ep['iteration'] for ep in ch_log]
    y1 = [ep['val_perplexity'] for ep in ch_log]
    y2 = [ep['perplexity'] for ep in ch_log]
    return x, y1, y2


def plot(x, y1, y2, title=None):
    lines = pyplot.plot(x, y1, label='val/perplexity')
    # ラインのスタイルを変更する
    # lines[0].set_color('#FF0000')  # 赤色に
    # lines[0].set_linestyle('-')
    # lines[0].set_linewidth(1)  # 線幅

    lines = pyplot.plot(x, y2, label='train/perplexity')

    pyplot.legend()  # 凡例表示
    pyplot.xlabel("iteration")
    pyplot.ylabel("perplexity")

    if title:
        pyplot.title(title)
    pyplot.grid(True)
    # 描画
    pyplot.show()

d = load_log('result/log')
plot(*d)
