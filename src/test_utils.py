# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import glob
import json
import os
import random as rnd
import shutil
import sys
from collections import defaultdict

import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow import keras
from tqdm import tqdm

from src.utils import euclidean_distance, loss


# -


class TestGenerator(keras.utils.Sequence):
    def __init__(
        self,
        data,
        img_size,
        batch_size=32,
        norm=False,
        n_chanels=1,
    ):
        self.data = data.reset_index(drop=True)
        self.img_size = img_size
        self.batch_size = batch_size
        self.norm = norm
        self.n_chanels = n_chanels
        self.default_img_size = (512, 81)

    def __len__(self):
        return self.data.shape[0] // self.batch_size

    def load_img(self, path):
        img = np.load(path).astype("float32")
        if self.norm:
            img -= img.min()
            img /= img.max()
        if self.img_size < self.default_img_size:
            wpad = (img.shape[1] - self.img_size[1]) // 2
            img = img[:, wpad : wpad + self.img_size[1]]
        if img.shape != self.img_size:
            wpad = self.img_size[1] - img.shape[1]
            wpad_l = wpad // 2
            wpad_r = wpad - wpad_l
            img = np.pad(
                img,
                pad_width=((0, 0), (wpad_l, wpad_r)),
                mode="constant",
                constant_values=0,
            )
        img = np.expand_dims(img, -1)
        if self.n_chanels == 3:
            img = np.concatenate([img, img, img], -1)
        return img

    def __getitem__(self, batch_ix):
        imgs = np.zeros([self.batch_size, *self.img_size, self.n_chanels])

        for i in range(self.batch_size):
            ix = i + self.batch_size * batch_ix
            img = self.load_img(self.data.loc[ix, "path"])
            imgs[i] = img

        return imgs

def embNet(
    path,
    input_shape=(512, 60, 1),
    dropout_rate=0.1,
    embedding_len=1024,
    activation_fn="relu",
):
    with open(os.path.join(path, "config.json"), "r") as f:
        params = json.load(f)
    act_fn = params["model"]["activation_fn"]
    embedding_len = params["model"]["embedding_len"]
    input_shape = params["model"]["input_shape"]
    base_model = keras.applications.MobileNet(
        input_shape=input_shape,
        alpha=1.0,
        depth_multiplier=1,
        dropout=dropout_rate,
        include_top=False,
        weights=None,
        pooling=None,
    )
    x = keras.layers.Flatten()(base_model.output)
    x = keras.layers.Dense(embedding_len * 4, activation=act_fn, name="dense_1")(x)
    x = keras.layers.Dense(embedding_len * 2, activation=act_fn, name="dense_2")(x)
    x = keras.layers.Dense(embedding_len, activation=None, name="dense_3")(x)
    # x = keras.layers.Reshape((1, -1))(x)
    embedding_net = keras.Model(inputs=base_model.input, outputs=x, name="embedding")
    weights = keras.models.load_model(
        os.path.join(path, "model.h5"),
        compile=False,
    )
    embedding_net.set_weights(weights.weights)
    return embedding_net


def get_emb_model(path):
    with open(os.path.join(path, "config.json"), "r") as f:
        params = json.load(f)
    act_fn = params["model"]["activation_fn"]
    kernel_size = params["model"]["kernel_size"]
    input = tf.keras.Input(
        shape=params["model"]["input_shape"], dtype="float32", name="imgs"
    )
    x = keras.layers.Conv2D(
        4,
        kernel_size,
        activation=act_fn,
        name="Conv2D_1",
    )(input)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), name=f"avg_pool_1")(x)
    x = keras.layers.Conv2D(
        filters=16,
        kernel_size=kernel_size,
        activation=act_fn,
        name="Conv2D_2",
    )(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), name=f"avg_pool_2")(x)

    x = keras.layers.Conv2D(
        filters=64,
        kernel_size=kernel_size,
        activation=act_fn,
        name="Conv2D_3",
    )(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), name=f"avg_pool_3")(x)

    x = keras.layers.Conv2D(
        filters=256,
        kernel_size=kernel_size,
        activation=act_fn,
        name="Conv2D_4",
    )(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), name=f"avg_pool_4")(x)

    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(
        params["model"]["embedding_len"],
        activation=act_fn,
        name=f"dense_{params['model']['embedding_len']}",
    )(x)
    embedding_net = keras.Model(inputs=input, outputs=x, name=f"embedding")
    weights = keras.models.load_model(
        os.path.join(path, "model.h5"), custom_objects={"contrastive_loss": loss(1)}
    )
    embedding_net.set_weights(weights.weights[:10])
    return embedding_net


def find_batch_size(data_size):
    div = []
    i = 1
    while i <= data_size:
        if data_size % i == 0:
            div.append(i)
        i += 1
    div = np.array(div)
    max_batch_size = np.max(np.where(div < 550, div, 0))
    return max_batch_size


def choose_100(
    prediction, df, val=True, path_to_save=os.getcwd(), file_ix=1, n_samples=100
):
    dists = pairwise_distances(prediction)
    neigh = {}
    with open(os.path.join(path_to_save, f"submission_{file_ix}"), "w") as f:
        for ix in tqdm(range(prediction.shape[0])):
            trackid = df.loc[ix, "trackid"]
            nearest_100 = np.argsort(dists[ix])[: n_samples + 1]
            tracks_100 = df.loc[nearest_100, "trackid"].tolist()
            neigh[trackid] = {"tracks": [x for x in tracks_100 if x != trackid]}
            if val:
                artist_100 = df.loc[nearest_100, "artistid"].tolist()
                neigh[trackid]["artists"] = artist_100
                neigh[trackid]["artistid"] = df.loc[ix, 'artistid']
            f.write(
                "{}\t{}\n".format(
                    trackid,
                    " ".join(list(map(str, tracks_100))),
                )
            )
    return neigh


def pairwise_distances(array):
    dists = tf.add(
        tf.reduce_sum(tf.square(array), axis=[1], keepdims=True),
        tf.reduce_sum(
            tf.square(tf.transpose(array)), axis=[0], keepdims=True
        ),
    ) - 2.0 * tf.matmul(array, tf.transpose(array))
    return dists

# +
def load_submission(input_path, max_top_size=100):
    result = {}
    with open(input_path, "r") as finput:
        for line in finput:
            query_trackid, answer_items = line.rstrip().split("\t", 1)
            query_trackid = int(query_trackid)
            ranked_list = []
            for result_trackid in answer_items.split(" "):
                result_trackid = int(result_trackid)
                if result_trackid != query_trackid:
                    ranked_list.append(result_trackid)
                if len(ranked_list) >= max_top_size:
                    break
            result[query_trackid] = ranked_list
    return result


def position_discounter(position):
    return 1.0 / np.log2(position + 1)


def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg


def compute_dcg(query_trackid, ranked_list, track2artist_map):
    query_artistid = track2artist_map[query_trackid]
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list):
        position = result_indx + 1
        discounted_position = position_discounter(position)
        result_artistid = track2artist_map[result_trackid]
        if result_artistid == query_artistid:
            dcg += discounted_position
    return dcg


def eval_submission(tracks_meta, submission, top_size = 100):
    track2artist_map = tracks_meta.set_index("trackid")["artistid"].to_dict()
    track2subset_map = tracks_meta.set_index("trackid")["subset"].to_dict()
    artist2tracks_map = tracks_meta.groupby("artistid").agg(list)["trackid"].to_dict()

    ndcg_list = defaultdict(list)
    for _, row in tracks_meta.iterrows():
        query_trackid = row["trackid"]
        ranked_list = submission.get(query_trackid, [])
        query_artistid = track2artist_map[query_trackid]
        query_artist_tracks_count = len(artist2tracks_map[query_artistid])
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count - 1, top_size=top_size)
        dcg = compute_dcg(query_trackid, ranked_list, track2artist_map)
        if ideal_dcg != 0:
            ndcg_list[track2subset_map[query_trackid]].append(dcg / ideal_dcg)

    result = {}
    for subset, values in ndcg_list.items():
        result[subset] = np.mean(values)
    return result
