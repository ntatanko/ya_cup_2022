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
import json
import os

import annoy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from src.data_utils import ImageLoader, TestLoader


# Losses
def triplet_loss(margin=0.2):
    def loss(y_true, y_pred):
        anchor, positive, negative = (
            y_pred[:, 0, ...],
            y_pred[:, 1, ...],
            y_pred[:, 2, ...],
        )
        positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
        return tf.maximum(positive_dist - negative_dist + margin, 0.0)

    return loss


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


def make_callbacks(
    path,
    monitor="val_loss",
    mode="min",
    reduce_patience=3,
    stop_patience=13,
    save_format=".h5",
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
            os.path.join(path, f"best{save_format}"),
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
    plt.show()


# Compute NDCG
def show_heatmap(df, cfg, model):
    def pairwise_distances(array):
        dists = (
            np.sum(np.square(array), axis=1, keepdims=True)
            + np.sum(np.square(np.transpose(array, (1, 0))), axis=0, keepdims=True)
            - 2.0 * np.matmul(array, np.transpose(array, (1, 0)))
        )
        return dists

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
    embeddings = model.predict(features_loader)
    plt.figure(figsize=(12, 10))
    plt.title("Pairwise distances")
    sns.heatmap(pairwise_distances(embeddings))
    plt.xticks([])
    plt.yticks([])
    plt.show()


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


# Save
def save_params(
    mod_dir,
    model,
    history,
    cfg,
    loss_name="contrastive_loss",
    metric="acc",
    triplets=False,
):
    train_history = history.history
    if not triplets:
        metric_name = "Max_accuracy"
        if metric == "loss":
            min_val_loss_ix = np.argmin(train_history["val_loss"])
            best_metric = train_history["val_accuracy"][min_val_loss_ix]
        elif metric == "acc":
            best_metric = np.max(train_history["val_accuracy"])
    else:
        metric_name = "Min_loss"
        best_metric = np.min(train_history["val_loss"])
    model.save(
        os.path.join(mod_dir, f"model_{int(np.round(best_metric * 1000))}.h5"),
        include_optimizer=False,
        save_traces=False,
    )
    for k in train_history.keys():
        train_history[k] = list(map(float, train_history[k]))
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
    path_to_save = os.path.join(
        mod_dir, f"model_{int(np.round(best_metric * 1000))}.h5"
    )
    print(f"\n{metric_name} = {best_metric}, model saved to {path_to_save}")


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
