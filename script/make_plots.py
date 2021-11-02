#!/usr/bin/env python3
from __future__ import annotations
from numpy import NaN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import os

BASE_DIRECTORY = "/home/mikolaj/pw/sn/projekt1"


def make_obj_path(problem, act, layers, kind, count):
    return "/home/mikolaj/pw/sn/projekt1/{}/result/objective/{}.{}.{}.{}.csv".format(problem, act, layers, kind, count)


def make_obj_out_path(problem, plot, act, layers, kind, count):
    try:
        os.makedirs("/home/mikolaj/pw/sn/projekt1/plots/{}/objective/{}".
                    format(problem, plot))
    except FileExistsError:
        pass
    return "/home/mikolaj/pw/sn/projekt1/plots/{}/objective/{}/{}.{}.{}.{}.pdf".format(problem, plot, act, layers, kind, count)


def make_lineplot(dt: pd.DataFrame, fields: list[str]) -> sbn.AxesSubplot:
    field_map = {'epoch': dt.epoch}
    for field in fields:
        field_map[field] = dt[field]

    dtf = pd.DataFrame(field_map)

    dtfm = pd.melt(dtf, ['epoch'])

    return sbn.lineplot(x='epoch', y='value', hue='variable', data=dtfm)


def make_lineplot_act(layer, kind, obs):
    dt_sig = pd.read_csv(make_obj_path(
        "regression", "sigmoid", layer, kind, obs))
    dt_tanh = pd.read_csv(make_obj_path(
        "regression", "tanh", layer, kind, obs))
    dt_gauss = pd.read_csv(make_obj_path(
        "regression", "gaussian", layer, kind, obs))
    return pd.concat([dt_sig, dt_tanh, dt_gauss], keys=[
        'sigmoid', 'tanh', 'gaussian'], names=['key'])


def make_lineplot_obs(layer, act, kind):
    dt_100 = pd.read_csv(make_obj_path(
        "regression", act, layer, kind, "100"))
    dt_500 = pd.read_csv(make_obj_path(
        "regression", act, layer, kind, "500"))
    dt_1000 = pd.read_csv(make_obj_path(
        "regression", act, layer, kind, "1000"))
    dt_10000 = pd.read_csv(make_obj_path(
        "regression", act, layer, kind, "10000"))
    return pd.concat([dt_100, dt_500, dt_1000, dt_10000], keys=[
        '100', '500', '1000', '10000'], names=['key'])


def make_lineplot_err(layer, act, kind, obs):
    dt = pd.read_csv(make_obj_path(
        "regression", act, layer, kind, obs))

    dt_fields = pd.DataFrame({
        'epoch': dt.epoch,
        'mse': dt.mse,
        'mae': dt.mae,
        # 'klasyfikacja (treningowy)': dt.train * 100,
        # 'klasyfikacja (testowy)': dt.test * 100,
        # "cross entropy (treningowy)": dt.train_cross * 100,
        # "cross entropy (testowy)": dt.train_cross * 100,
    })

    return pd.melt(dt_fields, ['epoch'])


def process_lineplot(dt: pd.DataFrame, name: str, fields: list[str], act: str, layer: str, kind: str, obs: str):
    fig = plt.figure()
    fig.clear()
    # train / test
    make_lineplot(dt, fields)
    plt.savefig(make_obj_out_path("classification", name,
                act, layer, kind, obs))
    plt.close(fig)


def main():
    dt = make_lineplot_err("none", "sigmoid", "activation", "1000")

    sbn.lineplot(x='epoch', y='value', hue='variable', data=dt)
    plt.show()

# dt_act = make_lineplot_obs("6-3-3-3", "sigmoid", "cube")
# print(dt_act)
# sbn.lineplot(x='epoch', y='mse', hue='key',
#              data=dt_act.replace(NaN, 1).where(dt_act.epoch < 2000))
# plt.show()

    return

    for act in ["sigmoid", "tanh", "gaussian"]:
        for layer in ["none", "4", "8-4", "8-8-4", "8-8-8-4"]:
            for kind in ["simple", "three_gauss"]:
                for obs in ["100", "500", "1000", "10000"]:
                    dt = pd.read_csv(make_obj_path(
                        "classification", act, layer, kind, obs))
                    process_lineplot(
                        dt, "class", ["train", "test"], act, layer, kind, obs)
                    process_lineplot(
                        dt, "entropy", ["train_cross", "test_cross"], act, layer, kind, obs)
                    process_lineplot(
                        dt, "mean", ["mse", "mae"], act, layer, kind, obs)


if __name__ == "__main__":
    main()
