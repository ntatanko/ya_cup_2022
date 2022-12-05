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
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.core.interactiveshell import InteractiveShell
from tensorflow import keras

InteractiveShell.ast_node_interactivity = "all"
from sklearn.model_selection import KFold
import annoy
import albumentations as A
from tqdm import tqdm


# -


def train_val_split(df, fold, n_splits=8, seed=42, data_dir = "/app/_data/artist_data/"):
    df["path"] = df["archive_features_path"].apply(
        lambda x: os.path.join(data_dir, "train_features", x)
    )
    gkf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for n, (train_artist_ids, val_artist_ids) in enumerate(
        gkf.split(
            X=df["artistid"].unique().tolist(),
        )
    ):
        df.loc[df.query('artistid in @val_artist_ids').index, 'fold'] = n
    train_df = df[df["fold"] != fold].reset_index(drop=True).copy()
    val_df = df[df["fold"] == fold].reset_index(drop=True).copy()
    return train_df, val_df



def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


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
            os.path.join(path, "best.h5"),
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode=mode,
            save_freq="epoch",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.7,
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


class ImageLoader:
    def __init__(
        self, target_size, augment=True, center_crop=False, norm=False, n_channels=None
    ):
        self.target_size = target_size
        self.augment = augment
        self.center_crop = center_crop
        self.norm = norm
        self.n_channels = n_channels

    def reshape_img(self, img):
        if img.shape[:2] < self.target_size:
            """img = A.PadIfNeeded(
                min_height=512,
                min_width=81,
                border_mode=3,
                always_apply=True,
            )(image=img)["image"]"""
            pad_width = self.target_size[1] - img.shape[1]
            pad_start = np.random.randint(img.shape[1] - pad_width)
            pad_chunk = img[:, pad_start : pad_start + pad_width]
            img = np.concatenate([img, pad_chunk], axis=1)
        elif img.shape[:2] > self.target_size:
            if self.center_crop:
                img = A.CenterCrop(
                    always_apply=True,
                    height=self.target_size[0],
                    width=self.target_size[1],
                )(image=img)["image"]
            else:
                img = A.RandomCrop(
                    always_apply=True,
                    height=self.target_size[0],
                    width=self.target_size[1],
                )(image=img)["image"]
        return img

    def augment_fn(self, img):
        transform = A.Compose(
            [
                A.Flip(p=0.2),
                A.PixelDropout(p=0.1, dropout_prob=0.01),
                A.CoarseDropout(
                    p=0.1,
                    max_holes=20,
                    max_height=5,
                    max_width=3,
                    min_holes=1,
                    min_height=2,
                    min_width=2,
                ),
                A.RandomGridShuffle(p=0.3, grid=(1, 6)),
            ]
        )
        return transform(image=img)["image"]

    def load_img(self, path):
        img = np.load(path)
        img = self.reshape_img(img)
        if self.norm:
            img -= img.min()
            img /= img.max()
        if self.augment:
            img = self.augment_fn(img)
        if self.n_channels == 1:
            img = np.expand_dims(img, -1)
        return img


class PairsGenerator(keras.utils.Sequence):
    def __init__(
        self,
        df,
        img_loader,
        batch_size=32,
        shuffle=True,
        positive_label=0,
        triplets=False,
    ):
        self.df = df
        self.img_loader = img_loader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.positive_label = positive_label
        self.negative_label = 0 if self.positive_label == 1 else 1
        self.artist_ids = self.df["artistid"].unique().tolist()
        self.artist2paths = self.df.groupby("artistid").agg(list)["path"].to_dict()
        if self.shuffle:
            np.random.shuffle(self.artist_ids)

    def __len__(self):
        return len(self.artist_ids) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.artist_ids)

    def make_pair(self, ix):
        artist_id = self.artist_ids[ix]
        if np.random.rand() > 0.5 and len(self.artist2paths[artist_id]) >= 2:
            path1, path2 = rnd.sample(self.artist2paths[artist_id], 2)
            label = self.positive_label
        else:
            path1 = rnd.sample(self.artist2paths[artist_id], 1)[0]
            new_artist_id = rnd.sample(
                [x for x in self.artist_ids if x != artist_id], 1
            )[0]
            path2 = rnd.sample(self.artist2paths[new_artist_id], 1)[0]
            label = self.negative_label
        imgs = [self.img_loader.load_img(x) for x in [path1, path2]]
        return imgs, label

    def __getitem__(self, batch_ix):
        X1, X2, Y = [], [], []
        for i in range(self.batch_size):
            (img1, img2), label = self.make_pair(ix=i + self.batch_size * batch_ix)
            X1.append(img1)
            X2.append(img2)
            Y.append(label)
        return {"img1": np.array(X1), "img2": np.array(X2)}, np.array(Y, dtype= 'float32')


class TestLoader(keras.utils.Sequence):
    def __init__(self, df, data_loader):
        self.df = df.reset_index(drop=True)
        self.data_loader = data_loader
        self.batch_size = 1
        self.track_ids = self.df["trackid"].tolist()
        self.track2path = self.df.set_index("trackid")["path"].to_dict()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, batch_ix):
        track_id = self.track_ids[batch_ix]
        img = self.data_loader.load_img(self.track2path[track_id])
        return np.expand_dims(img, 0), track_id


def tf_inference(model, loader):
    track_ids = loader.track_ids
    tracks_embeds = model.predict(loader)
    embeds = {k: v for k, v in zip(track_ids, tracks_embeds)}
    return embeds


def get_ranked_list(embeds, top_size=100, annoy_num_trees=128, annoy_metric="angular"):
    annoy_index = None
    annoy2id = []
    id2annoy = dict()
    for track_id, track_embed in tqdm(embeds.items()):
        id2annoy[track_id] = len(annoy2id)
        annoy2id.append(track_id)
        if annoy_index is None:
            annoy_index = annoy.AnnoyIndex(len(track_embed), annoy_metric)
        annoy_index.add_item(id2annoy[track_id], track_embed)
    annoy_index.build(annoy_num_trees, n_jobs=-1)
    ranked_list = dict()
    for track_id in tqdm(embeds.keys()):
        candidates = annoy_index.get_nns_by_item(id2annoy[track_id], top_size + 1)[1:]
        candidates = list(filter(lambda x: x != id2annoy[track_id], candidates))
        ranked_list[track_id] = [annoy2id[candidate] for candidate in candidates]
    return ranked_list


def position_discounter(position):
    return 1.0 / np.log2(position + 1)


def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg


def compute_dcg(query_trackid, ranked_list, track2artist, top_size):
    query_artistid = track2artist[query_trackid]
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list[:top_size]):
        assert result_trackid != query_trackid
        position = result_indx + 1
        discounted_position = position_discounter(position)
        result_artistid = track2artist[result_trackid]
        if result_artistid == query_artistid:
            dcg += discounted_position
    return dcg


def eval_submission(submission, val_df, top_size=100):
    track2artist = val_df.set_index("trackid")["artistid"].to_dict()
    artist2tracks = val_df.groupby("artistid").agg(list)["trackid"].to_dict()
    ndcg_list = []
    for query_trackid in tqdm(submission.keys()):
        ranked_list = submission[query_trackid]
        query_artistid = track2artist[query_trackid]
        query_artist_tracks_count = len(artist2tracks[query_artistid])
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count - 1, top_size=top_size)
        dcg = compute_dcg(query_trackid, ranked_list, track2artist, top_size=top_size)
        try:
            ndcg_list.append(dcg / ideal_dcg)
        except ZeroDivisionError:
            continue
    return np.mean(ndcg_list)


def compute_val_dcg(model, df, cfg, annoy_metric="angular"):
    features_loader = TestLoader(
        df=df,
        data_loader=ImageLoader(
            target_size=cfg.img_size,
            augment=False,
            center_crop=cfg.center_crop,
            norm=cfg.norm,
            n_channels=cfg.n_channels,
        ),
    )
    print("Making prediction")
    embeds = tf_inference(model, features_loader)
    print("\nComputing ranked list")
    ranked_list = get_ranked_list(
        embeds=embeds, top_size=100, annoy_num_trees=128, annoy_metric=annoy_metric
    )
    print("\nCalculating NDCG")
    val_ndcg = eval_submission(
        ranked_list,
        df,
        top_size=100,
    )
    print(f"\nNDCG on val set = {val_ndcg:.5f}")

def save_params(
    mod_dir, model, history, cfg, loss_name="contrastive_loss", metric="acc"
):
    train_history = history.history
    if metric == "loss":
        min_val_loss_ix = np.argmin(train_history["val_loss"])
        max_acc = train_history["val_accuracy"][min_val_loss_ix]
    elif metric == "acc":
        max_acc = np.max(train_history["val_accuracy"])
    for k in train_history.keys():
        train_history[k] = list(map(float, train_history[k]))
    model.save(
        os.path.join(mod_dir, f"model_{int(np.round(max_acc * 1000))}.h5"),
        include_optimizer=False,
        save_traces=False,
    )
    config = {
        "loss": loss_name,
        "pos_label": cfg.pos_label,
        "history": train_history,
        "norm": cfg.norm,
        "fold": cfg.fold,
        "model": {
            "input_shape": cfg.input_shape,
            "embedding_len": cfg.emb_len,
            "kernel_size": cfg.kernel_size,
            "activation_fn": cfg.act_fn,
            "batch_norm": cfg.batch_norm,
            "n_channels": cfg.n_channels,
            "center_crop": cfg.center_crop,
        },
    }
    with open(os.path.join(mod_dir, "config.json"), "w") as f:
        json.dump(config, f)
    print(
        f'\nMax_acc = {max_acc}, model saved to {os.path.join(mod_dir, f"model_{int(np.round(max_acc * 1000))}.h5")}'
    )


def save_submission(
    model,
    test_df,
    cfg,
    submission_path,
    annoy_metric="angular",
    top_size=100,
    annoy_num_trees=256,
):
    submission_path = os.path.join(submission_path, f"submission_{top_size}.txt")
    test_df["path"] = test_df["archive_features_path"].apply(
        lambda x: os.path.join("/app/_data/artist_data/", "test_features", x)
    )
    features_loader = TestLoader(
        df=test_df,
        data_loader=ImageLoader(
            target_size=cfg.img_size,
            augment=False,
            center_crop=cfg.center_crop,
            norm=cfg.norm,
            n_channels=cfg.n_channels,
        ),
    )
    print("Making prediction")
    embeds = tf_inference(model, features_loader)
    print("\nComputing ranked list")
    ranked_list = get_ranked_list(
        embeds=embeds,
        top_size=top_size,
        annoy_num_trees=annoy_num_trees,
        annoy_metric=annoy_metric,
    )

    with open(submission_path, "w") as f:
        for query_trackid, result in ranked_list.items():
            f.write(
                "{}\t{}\n".format(query_trackid, " ".join(map(str, result[:top_size])))
            )
    print(f"\nSubmission file saved to {submission_path}")


