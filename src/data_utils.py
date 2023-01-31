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

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from tensorflow import keras


def train_val_split(df, fold, n_splits=8, seed=42, data_dir="/app/_data/artist_data/"):
    df["path"] = df["archive_features_path"].apply(
        lambda x: os.path.join(data_dir, "train_features", x)
    )
    gkf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for n, (train_artist_ids, val_artist_ids) in enumerate(
        gkf.split(
            X=df["artistid"].unique().tolist(),
        )
    ):
        df.loc[df.query("artistid in @val_artist_ids").index, "fold"] = n
    train_df = df[df["fold"] != fold].reset_index(drop=True).copy()
    val_df = df[df["fold"] == fold].reset_index(drop=True).copy()
    return train_df, val_df


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


class SemiHardTripletsGenerator(keras.utils.Sequence):
    def __init__(
        self,
        df,
        data_loader,
        batch_size=32,
        shuffle=True,
    ):
        self.df = df.reset_index(drop=True)
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.artist_ids = self.df["artistid"].unique().tolist()
        if self.shuffle:
            np.random.shuffle(self.artist_ids)
        self.artist2path = self.df.groupby("artistid").agg(list)["path"].to_dict()
        self.path2artist = self.df.set_index("path")["artistid"].to_dict()
        self.artist_groups = self.group_artist()

    def __len__(self):
        return len(self.artist_groups)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.artist_ids)

    def group_artist(self):
        groups, batch = [], []
        for artist in self.artist_ids:
            batch.extend(self.artist2path[artist])
            if len(batch) >= self.batch_size:
                np.random.shuffle(batch)
                groups.append(batch[: self.batch_size])
                batch = []
        return groups

    def __getitem__(self, batch_ix):
        batch_paths = self.artist_groups[batch_ix]
        imgs, labels = [], []
        for i in range(self.batch_size):
            imgs.append(self.data_loader.load_img(batch_paths[i]))
            labels.append(self.path2artist[batch_paths[i]])
        imgs = np.array(imgs)
        labels = np.array(labels)
        return imgs, labels


class TripletsGenerator(keras.utils.Sequence):
    def __init__(
        self,
        df,
        data_loader,
        batch_size=32,
        shuffle=True,
    ):
        self.df = (
            df.merge(
                df["artistid"].value_counts(),
                left_on="artistid",
                right_index=True,
                suffixes=[None, "_count"],
            )
            .query("artistid_count >=2")
            .reset_index(drop=True)
        )
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.artist_ids = self.df["artistid"].unique().tolist()
        if self.shuffle:
            np.random.shuffle(self.artist_ids)
        self.artis2path = self.df.groupby("artistid").agg(list)["path"].to_dict()
        self.triplets = self.make_triplets()

    def __len__(self):
        return len(self.triplets)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.artist_ids)
        self.triplets = self.make_triplets()

    def make_triplets(self):
        batch, triplets = [], []
        for artist in self.artist_ids:
            anchor_path, positive_path = rnd.sample(self.artis2path[artist], 2)
            neg_artist = rnd.sample([x for x in self.artist_ids if x != artist], 1)[0]
            negative_path = rnd.sample(self.artis2path[neg_artist], 1)[0]
            batch.append([anchor_path, positive_path, negative_path])
            if len(batch) == self.batch_size:
                triplets.append(batch)
                batch = []
        return triplets

    def __getitem__(self, batch_ix):
        imgs = {
            "anchor": [],
            "positive": [],
            "negative": [],
        }
        for paths in self.triplets[batch_ix]:
            imgs["anchor"].append(self.data_loader.load_img(paths[0]))
            imgs["positive"].append(self.data_loader.load_img(paths[1]))
            imgs["negative"].append(self.data_loader.load_img(paths[2]))
        imgs["anchor"] = np.array(imgs["anchor"])
        imgs["positive"] = np.array(imgs["positive"])
        imgs["negative"] = np.array(imgs["negative"])
        return imgs, np.zeros(self.batch_size)


def plot_triplets(triplets_generator, n_examples=5):
    path2artist = triplets_generator.df.set_index("path")["artistid"].to_dict()
    triplets = triplets_generator.triplets[
        np.random.randint(0, len(triplets_generator.triplets))
    ][:n_examples]
    plt.figure(figsize=(15, 2 * n_examples))
    for i in range(n_examples):
        anc, pos, neg = triplets[i]
        artist_ids = [path2artist[p] for p in [anc, pos, neg]]
        imgs = [np.load(p) for p in [anc, pos, neg]]
        for j in range(3):
            plt.subplot(n_examples, 3, 3 * i + j + 1)
            plt.title(f"artist {artist_ids[j]}")
            plt.imshow(imgs[j].transpose(1, 0))
            plt.xticks([])
            plt.yticks([])
    plt.show()


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
        return {"img1": np.array(X1), "img2": np.array(X2)}, np.array(
            Y, dtype="float32"
        )


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
