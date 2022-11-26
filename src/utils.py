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
import os
import random as rnd

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.core.interactiveshell import InteractiveShell
from tensorflow import keras

InteractiveShell.ast_node_interactivity = "all"


# -

class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        data,
        img_size,
        batch_size=32,
        norm=False,
        n_chanels=1,
        shuffle=True,
        positive_label=0,
        triplets = False
    ):
        self.data = data
        self.img_size = img_size
        self.batch_size = batch_size
        self.norm = norm
        self.n_chanels = n_chanels
        self.shuffle = shuffle
        self.positive_label = positive_label
        self.negative_label = 0 if self.positive_label == 1 else 1
        self.artist_ids = [x for x in self.data.keys()]
        self.default_img_size = (512, 81)
        self.triplets = triplets
        if self.shuffle:
            np.random.shuffle(self.artist_ids)

    def __len__(self):
        return len(self.artist_ids) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.artist_ids)

    def load_img(self, path):
        img = np.load(path).astype("float32")
        if self.norm:
            img -= img.min()
            img /= img.max()
        if self.img_size < self.default_img_size:
            wpad = (img.shape[1] - self.img_size[1])//2
            img = img[:, wpad: wpad + self.img_size[1]]
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

    def make_pair(self, ix, same_artist=True):
        artist_id = self.artist_ids[ix]
        if self.data[artist_id]["count"] < 2:
            same_artist = False
        if same_artist:
            path1, path2 = rnd.sample(self.data[artist_id]["paths"], 2)
        else:
            path1 = rnd.sample(self.data[artist_id]["paths"], 1)[0]
            new_artist_id = artist_id
            while artist_id == new_artist_id:
                new_artist_id = rnd.sample(self.artist_ids, 1)[0]
                path2 = rnd.sample(self.data[new_artist_id]["paths"], 1)[0]
        return same_artist, (path1, path2)

    def _get_one(self, ix, same_artist):
        upd_same_artist, (path1, path2) = self.make_pair(
            ix=ix,
            same_artist=same_artist
        )
        img1 = self.load_img(path1)
        img2 = self.load_img(path2)
        y = self.positive_label if upd_same_artist else self.negative_label
        return (img1, img2), y

    def __getitem__(self, batch_ix):
        b_X1 = np.zeros(
            (self.batch_size, self.img_size[0], self.img_size[1], self.n_chanels),
            dtype=np.float32,
        )
        b_X2 = np.zeros(
            (self.batch_size, self.img_size[0], self.img_size[1], self.n_chanels),
            dtype=np.float32,
        )
        b_Y = np.zeros(
            self.batch_size,
            dtype=np.float32,
        )
        for i in range(self.batch_size):
            (b_X1[i], b_X2[i]), b_Y[i] = self._get_one(
                ix=i + self.batch_size * batch_ix,
                same_artist=np.random.random() > 0.5
            )

        return {"img1": b_X1, "img2": b_X2}, b_Y


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def construct_embedding_model(
    input,
    embedding_len=1024,
    n_blocks=4,
    kernel_size=(10, 3),
    activation_fn="relu",
    batch_norm=False,
):
    depth_vector = 2 ** ((np.arange(n_blocks) + 1) * 2)

    def base_block(x, i):
        x = keras.layers.Conv2D(
            filters=depth_vector[i],
            kernel_size=kernel_size,
            activation=activation_fn,
            name=f"Conv2D_{i + 1}",
        )(x)
        x = keras.layers.AveragePooling2D(pool_size=(2, 2), name=f"avg_pool_{i + 1}")(x)
        return x

    if batch_norm:
        x = keras.layers.BatchNormalization()(input)
    else:
        x = input
    for i in range(n_blocks):
        x = base_block(x, i)
    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(
        embedding_len, activation=activation_fn, name=f"dense_{embedding_len}"
    )(x)
    embedding_net = keras.Model(inputs=input, outputs=x, name=f"embedding")
    return embedding_net


def make_model(
    input_shape=(512, 81, 1),
    n_blocks=4,
    kernel_size=(10, 3),
    embedding_len=1024,
    activation_fn="relu",
    batch_norm=False,
):
    base_model = construct_embedding_model(
        keras.layers.Input(input_shape),
        embedding_len=embedding_len,
        n_blocks=n_blocks,
        kernel_size=kernel_size,
        activation_fn=activation_fn,
        batch_norm=batch_norm,
    )

    input_1 = keras.layers.Input(input_shape, name="img1")
    input_2 = keras.layers.Input(input_shape, name="img2")
    node1 = base_model(input_1)
    node2 = base_model(input_2)

    merge_layer = keras.layers.Lambda(euclidean_distance)([node1, node2])
    output_layer = keras.layers.Dense(1, activation="sigmoid")(merge_layer)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    return siamese


def loss(margin=1):
    # from https://keras.io/examples/vision/siamese_contrastive/#define-the-constrastive-loss
    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


def plot_history(history):
    tran_keys = [k for k, v in history.items() if "val" not in k and k != "lr"]
    n_plots = len(tran_keys)
    n_rows = int(np.ceil(n_plots / 2))
    plt.figure(figsize=(20, 5 * n_rows))
    plt.suptitle("Training history")
    for n in range(n_plots):
        label = tran_keys[n]
        plt.subplot(n_rows, 2, n + 1)
        plt.title(label)
        plt.plot(history[label], label=f"train_{label}")
        plt.plot(history[f"val_{label}"], label=f"val_{label}")
        plt.legend()
    plt.show();


def make_callbacks(
    path, monitor="val_loss", mode="min", reduce_patience=10, stop_patience=100
):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=stop_patience,
            restore_best_weights=True,
            verbose=1,
            mode=mode,
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(path, "best"),
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode=mode,
            save_freq="epoch",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.9,
            patience=reduce_patience,
            verbose=1,
            mode=mode,
            min_delta=1e-4,
            min_lr=0.00000001,
        ),
        keras.callbacks.TensorBoard(
            log_dir=f"/app/.tensorboard/{path.split('/')[-2]}/", histogram_freq=0
        ),
        keras.callbacks.BackupAndRestore(os.path.join(path, "backup")),
        keras.callbacks.TerminateOnNaN(),
    ]
    return callbacks


class TripletsGenerator(keras.utils.Sequence):
    def __init__(
        self,
        data,
        img_size,
        batch_size=32,
        norm=False,
        n_chanels=1,
        shuffle=True,
        debug=True,
    ):
        self.data = data.reset_index(drop=True)
        self.img_size = img_size
        self.batch_size = batch_size
        self.norm = norm
        self.n_chanels = n_chanels
        self.shuffle = shuffle
        self.artist_ids = self.data["artistid"].unique().tolist()
        self.default_img_size = (512, 81)
        self.artis2path = self.data.groupby("artistid").agg(list)["path"].to_dict()
        self.paths = self.data["path"].tolist()
        self.debug = debug
        if self.shuffle:
            np.random.shuffle(self.paths)

    def __len__(self):
        return self.data.shape[0] // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.artist_ids)

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

    def make_triplet(self):
        art_pos, art_neg = rnd.sample(self.artist_ids, 2)
        anchor_path, positive_path = rnd.sample(self.artis2path[art_pos], 2)
        negative_path = rnd.sample(self.artis2path[art_neg], 1)[0]
        anchor, positive, negative = [
            self.load_img(path) for path in [anchor_path, positive_path, negative_path]
        ]
        if self.debug:
            return (
                (anchor, positive, negative),
                (anchor_path, positive_path, negative_path),
                (art_pos, art_neg),
            )
        else:
            return (anchor, positive, negative)

    def __getitem__(self, batch_ix):
        imgs = {
            "anchor": np.zeros([self.batch_size, *self.img_size, self.n_chanels]),
            "positive": np.zeros([self.batch_size, *self.img_size, self.n_chanels]),
            "negative": np.zeros([self.batch_size, *self.img_size, self.n_chanels]),
        }
        for i in range(self.batch_size):
            (
                imgs["anchor"][i, ...],
                imgs["positive"][i, ...],
                imgs["negative"][i, ...],
            ) = self.make_triplet()

        return imgs, np.zeros(self.batch_size)


def plot_triplets(df, cfg, n_examples=5):
    path2artist = df.set_index("path")["artistid"].to_dict()
    generator = TripletsGenerator(
        data=df,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        norm=cfg.norm,
        n_chanels=cfg.n_chanels,
        shuffle=True,
        debug=True,
    )
    plt.figure(figsize=(20, 2 * n_examples))
    for i in range(n_examples):
        imgs, paths, arts = generator.make_triplet()
        artist_ids = [path2artist[p] for p in paths]
        for j in range(3):
            plt.subplot(n_examples, 3, 3 * i + j + 1)
            plt.title(f"artist {artist_ids[j]}")
            plt.imshow(np.squeeze(imgs[j], -1).transpose(1, 0))
            plt.xticks([])
            plt.yticks([])
    plt.show()


# class DataGenerator(keras.utils.Sequence):
#     def __init__(
#         self,
#         data,
#         img_size,
#         batch_size=32,
#         norm=False,
#         n_chanels=1,
#         shuffle=True,
#         positive_label=0,
#         triplets=False,
#     ):
#         self.data = data
#         self.img_size = img_size
#         self.batch_size = batch_size
#         self.norm = norm
#         self.n_chanels = n_chanels
#         self.shuffle = shuffle
#         self.positive_label = positive_label
#         self.negative_label = 0 if self.positive_label == 1 else 1
#         self.artist_ids = [x for x in self.data.keys()]
#         self.default_img_size = (512, 81)
#         self.triplets = triplets
#         if self.shuffle:
#             np.random.shuffle(self.artist_ids)

#     def __len__(self):
#         return len(self.artist_ids) // self.batch_size

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.artist_ids)

#     def load_img(self, path):
#         img = np.load(path).astype("float32")
#         if self.norm:
#             img -= img.min()
#             img /= img.max()
#         if self.img_size < self.default_img_size:
#             wpad = (img.shape[1] - self.img_size[1]) // 2
#             img = img[:, wpad : wpad + self.img_size[1]]
#         if img.shape != self.img_size:
#             wpad = self.img_size[1] - img.shape[1]
#             wpad_l = wpad // 2
#             wpad_r = wpad - wpad_l
#             img = np.pad(
#                 img,
#                 pad_width=((0, 0), (wpad_l, wpad_r)),
#                 mode="constant",
#                 constant_values=0,
#             )
#         img = np.expand_dims(img, -1)
#         if self.n_chanels == 3:
#             img = np.concatenate([img, img, img], -1)
#         return img

#     def make_pair(self, ix, same_artist=True):
#         artist_id = self.artist_ids[ix]
#         if self.data[artist_id]["count"] < 2:
#             same_artist = False
#         if same_artist:
#             path1, path2 = rnd.sample(self.data[artist_id]["paths"], 2)
#         else:
#             path1 = rnd.sample(self.data[artist_id]["paths"], 1)[0]
#             new_artist_id = artist_id
#             while artist_id == new_artist_id:
#                 new_artist_id = rnd.sample(self.artist_ids, 1)[0]
#                 path2 = rnd.sample(self.data[new_artist_id]["paths"], 1)[0]
#         return same_artist, (path1, path2)

#     def make_triplet(self, ix):
#         artist_id = self.artist_ids[ix]
#         if self.data[artist_id]["count"] >= 2:
#             paths = rnd.sample(self.data[artist_id]["paths"], 2)
#         elif self.data[artist_id]["count"] == 1:
#             paths = self.data[artist_id]["paths"]
#         if len(paths) == 1:
#             n_neg_samples = 2
#             labels = [0, 0]
#         else:
#             n_neg_samples = 1
#             labels = [1, 0]
#         new_artist_ids = [artist_id]
#         while artist_id in new_artist_ids:
#             new_artist_ids = rnd.sample(self.artist_ids, n_neg_samples)
#             neg_paths = [
#                 rnd.sample(self.data[new_artist_id]["paths"], 1)[0]
#                 for new_artist_id in new_artist_ids
#             ]
#         paths.extend(neg_paths)
#         if np.random.random() > 0.5:
#             paths = [paths[0], paths[2], paths[1]]
#             labels = [labels[1], labels[0]]
#         return labels, paths

#     def _get_one(self, ix, same_artist):
#         if self.triplets:
#             labels, paths = self.make_triplet(ix)
#             imgs = [self.load_img(path) for path in paths]
#             y = labels
#         else:
#             upd_same_artist, paths = self.make_pair(ix=ix, same_artist=same_artist)
#             imgs = [self.load_img(path) for path in paths]
#             y = self.positive_label if upd_same_artist else self.negative_label
#         return np.array(imgs), y

#     def __getitem__(self, batch_ix):
#         n_imgs = 3 if self.triplets else 2
#         n_labels = 2 if self.triplets else 1
#         b_X = np.zeros(
#             (
#                 self.batch_size,
#                 n_imgs,
#                 self.img_size[0],
#                 self.img_size[1],
#                 self.n_chanels,
#             ),
#             dtype=np.float32,
#         )
#         b_Y = np.zeros(
#             (self.batch_size, n_labels),
#             dtype=np.float32,
#         )
#         for i in range(self.batch_size):
#             b_X[i], b_Y[i] = self._get_one(
#                 ix=i + self.batch_size * batch_ix, same_artist=np.random.random() > 0.5
#             )
#         # return b_X, b_Y
#         return {f"img{i}": b_X[:, i - 1, :, :] for i in range(1, n_imgs + 1)}, b_Y