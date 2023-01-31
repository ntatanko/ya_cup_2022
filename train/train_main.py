import json
import os
import random
from argparse import ArgumentParser

import albumentations as A
import annoy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.data_utils import ImageLoader, train_val_split
from src.utils import eval_submission, get_ranked_list


class TrainLoader:
    def __init__(self, df, features_loader, batch_size=256, device="cpu"):
        self.df = df
        self.features_loader = features_loader
        self.batch_size = batch_size
        self.device = device
        self.artist_track_ids = self.df.groupby("artistid").agg(list)
        self.track2path = self.df.set_index("trackid")["path"].to_dict()

    def _generate_pairs(self, track_ids):
        np.random.shuffle(track_ids)
        pairs = [track_ids[i - 2 : i] for i in range(2, len(track_ids) + 1, 2)]
        return pairs

    def _get_pair_ids(self):
        pairs = []
        artist_track_ids = self.artist_track_ids.copy()
        artist_track_pairs = artist_track_ids["trackid"].map(self._generate_pairs)
        for pair_ids in artist_track_pairs.explode().dropna():
            pairs.append(pair_ids)
        np.random.shuffle(pairs)
        return pairs

    def load_batch(self, tracks_ids):
        batch = [
            self.features_loader.load_img(self.track2path[track_id])
            for track_id in tracks_ids
        ]
        return torch.tensor(np.array(batch)).to(self.device)

    def _get_batch(self, batch_ids):
        batch_ids = np.array(batch_ids).reshape(-1)
        batch_features = self.load_batch(batch_ids)
        batch_features = batch_features.reshape(
            self.batch_size, 2, *self.features_loader.target_size
        )
        return batch_features

    def __iter__(self):
        batch_ids = []
        for pair_ids in self._get_pair_ids():
            batch_ids.append(pair_ids)
            if len(batch_ids) == self.batch_size:
                batch = self._get_batch(batch_ids)
                yield batch
                batch_ids = []

    def _len(self):
        return sum(1 for x in self._get_pair_ids()) // self.batch_size


class TestLoader:
    def __init__(self, df, features_loader, batch_size=256, device="cpu"):
        self.df = df
        self.features_loader = features_loader
        self.batch_size = batch_size
        self.device = device
        if "path" not in self.df.columns:
            self.df["path"] = self.df["archive_features_path"].apply(
                lambda x: os.path.join("/app/_data/artist_data/", "test_features", x)
            )
        self.track2path = self.df.set_index("trackid")["path"].to_dict()

    def load_batch(self, paths):
        batch = [self.features_loader.load_img(path) for path in paths]
        return torch.tensor(np.array(batch)).to(self.device)

    def __iter__(self):
        batch_ids, paths = [], []
        for track_id in tqdm(self.df.trackid.tolist()):
            path = self.track2path[track_id]
            batch_ids.append(track_id)
            paths.append(path)
            if len(batch_ids) == self.batch_size:
                yield batch_ids, self.load_batch(paths)
                batch_ids, paths = [], []
        if len(batch_ids) > 0:
            yield batch_ids, self.load_batch(paths)


class Custom_Loss(nn.Module):
    def __init__(self, temperature):
        super(Custom_Loss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_fn = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, y_i, y_j):
        batch_size = y_i.shape[0]
        N = 2 * batch_size
        z = torch.cat((y_i, y_j), dim=0)
        sim = self.similarity_fn(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        labels[0] = 1
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class BaseModel(nn.Module):
    def __init__(
        self,
        img_size,
        output_features_size1=512,
        output_features_size2=128,
        embed_len=128,
        kernel_size=3,
    ):
        super().__init__()
        self.output_features_size = output_features_size1 + output_features_size2
        self.embed_len = embed_len
        self.img_size = img_size
        self.conv_1 = nn.Conv1d(
            self.img_size[0], output_features_size1, kernel_size=kernel_size, padding=1
        )
        self.conv_2 = nn.Conv1d(
            output_features_size1,
            output_features_size1,
            kernel_size=kernel_size,
            padding=1,
        )
        self.mp_1 = nn.MaxPool1d(2)
        self.conv_1_2 = nn.Conv1d(
            self.img_size[1], output_features_size2, kernel_size=kernel_size, padding=1
        )
        self.conv_2_2 = nn.Conv1d(
            output_features_size2,
            output_features_size2,
            kernel_size=kernel_size,
            padding=1,
        )
        self.mp_1_2 = nn.MaxPool1d(2)
        self.linear = nn.Sequential(
            nn.Linear(self.output_features_size, self.output_features_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.output_features_size, self.embed_len, bias=False),
            nn.ReLU(),
            nn.Linear(self.embed_len, self.embed_len, bias=False),
        )

    def forward(self, x):
        y = x.transpose(2, 1)
        x = nn.ReLU()(self.conv_1(x))
        x = nn.ReLU()(self.conv_2(x))
        x = self.mp_1(x).mean(axis=2)

        y = nn.ReLU()(self.conv_1_2(y))
        y = nn.ReLU()(self.conv_2_2(y))
        y = self.mp_1_2(y).mean(axis=2)
        output = torch.cat([x, y], axis=-1)
        # output = F.normalize(output, p=2.0, dim=1)
        output = self.linear(output)

        return output


def inference(model, loader):
    embeds = dict()
    model.eval()
    for tracks_ids, tracks_features in loader:
        with torch.no_grad():
            tracks_embeds = model(tracks_features)
            for track_id, track_embed in zip(tracks_ids, tracks_embeds):
                embeds[track_id] = track_embed.cpu().numpy()
    return embeds


class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_metric = 0

    def early_stop(self, validation_metric):
        if validation_metric > self.min_validation_metric:
            self.min_validation_metric = validation_metric
            self.counter = 0
        elif validation_metric < (self.min_validation_metric + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(
    model,
    train_loader,
    val_loader,
    val_df,
    optimizer,
    criterion,
    num_epochs,
    model_path,
    history_path,
    top_size=100,
    sceduler_patience=4,
    early_stopping_patience=13,
):
    max_ndcg = None
    best_epoch = 0
    metrics = {"train_loss": [], "ndcg": []}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=sceduler_patience, verbose=True
    )
    early_stopper = EarlyStopper(patience=early_stopping_patience)
    for epoch in range(num_epochs):
        train_bar = tqdm(
            enumerate(train_loader),
            total=train_loader._len(),
            desc=f"Train: epoch #{epoch + 1}/{num_epochs}",
        )
        running_loss = 0
        epoch_loss = 0
        for n, batch in train_bar:
            optimizer.zero_grad()
            model.train()
            x_i, x_j = batch[:, 0, :, :], batch[:, 1, :, :]
            y_i = model(x_i)
            y_j = model(x_j)
            loss = criterion(y_i, y_j)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss = running_loss / (n + 1)
            current_lr = optimizer.param_groups[0]["lr"]
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            train_bar.set_postfix(
                train_loss=f"{epoch_loss:0.4f}",
                lr=f"{current_lr:0.6f}",
                gpu_memory=f"{memory:0.2f} GB",
            )
        metrics["train_loss"].append(epoch_loss)

        with torch.no_grad():
            print("\nValidation nDCG on epoch {}".format(epoch + 1))
            embeds = inference(model, val_loader)
            ranked_list = get_ranked_list(embeds, top_size, 128, "angular")
            val_ndcg = eval_submission(ranked_list, val_df)
            metrics["ndcg"].append(val_ndcg)
            print(f"ndcg = {val_ndcg} at {epoch + 1} epoch")
            if (max_ndcg is None) or (val_ndcg > max_ndcg):
                print(
                    f"ndcg improved from {max_ndcg} to {val_ndcg}, checkpoints saved to {model_path}"
                )
                max_ndcg = val_ndcg
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_path)

            scheduler.step(val_ndcg)
            if early_stopper.early_stop(val_ndcg):
                break
    with open(os.path.join(history_path, "history.json"), "w") as f:
        json.dump(metrics, f)
    print(f"Best ndcg = {max_ndcg} at {best_epoch} epoch\n")


def save_submission(submission, submission_path, top=100):
    with open(submission_path, "w") as f:
        for query_trackid, result in submission.items():
            f.write("{}\t{}\n".format(query_trackid, " ".join(map(str, result[:top]))))


def main():
    parser = ArgumentParser(description="Simple naive baseline")
    parser.add_argument(
        "--base-dir",
        dest="base_dir",
        action="store",
        required=False,
        type=str,
        nargs="?",
        const="/app/_data/artist_data/",
        default="/app/_data/artist_data/",
    )
    parser.add_argument(
        "--save-dir", dest="save_dir", action="store", required=True, type=str
    )
    parser.add_argument(
        "--fold",
        dest="fold",
        action="store",
        required=False,
        nargs="?",
        const=0,
        type=int,
        default=0,
    )
    parser.add_argument(
        "--img-size",
        dest="img_size",
        action="store",
        required=False,
        nargs="?",
        const=(512, 81),
        type=int,
        default=(512, 81),
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        action="store",
        required=False,
        nargs="?",
        const=512,
        type=int,
        default=512,
    )
    parser.add_argument(
        "--n-channels",
        dest="n_channels",
        action="store",
        required=False,
        nargs="?",
        const=256,
        type=int,
        default=256,
    )
    parser.add_argument(
        "--emb-len",
        dest="emb_len",
        action="store",
        required=False,
        nargs="?",
        const=128,
        type=int,
        default=128,
    )
    parser.add_argument(
        "--kernel-size",
        dest="kernel_size",
        action="store",
        required=False,
        nargs="?",
        const=3,
        type=int,
        default=3,
    )
    parser.add_argument(
        "--n-epochs",
        dest="n_epochs",
        action="store",
        required=False,
        nargs="?",
        const=150,
        type=int,
        default=150,
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        action="store",
        required=False,
        nargs="?",
        const=0.01,
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--stopping-patience",
        dest="early_stopping_patience",
        action="store",
        required=False,
        nargs="?",
        const=13,
        type=int,
        default=13,
    )
    parser.add_argument(
        "--sceduler-patience",
        dest="sceduler_patience",
        action="store",
        required=False,
        nargs="?",
        const=13,
        type=int,
        default=13,
    )
    args = parser.parse_args()

    # Seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size

    LR = 1e-4
    TRAINSET_PATH = os.path.join(args.base_dir, "train_features")
    TESTSET_PATH = os.path.join(args.base_dir, "test_features")
    TRAINSET_META_PATH = os.path.join(args.base_dir, "train_meta.tsv")
    TESTSET_META_PATH = os.path.join(args.base_dir, "test_meta.tsv")
    BASE_DIR = os.path.join(args.base_dir, "models", args.save_dir)
    SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.txt")
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "best.pt")

    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    with open(
        os.path.join(args.base_dir, "models", args.save_dir, "args.json"), "w"
    ) as f:
        json.dump(args.__dict__, f)

    model = BaseModel(
        IMG_SIZE, args.n_channels, 128, args.emb_len, args.kernel_size
    ).to(DEVICE)

    df = pd.read_csv(TRAINSET_META_PATH, sep="\t")
    test_df = pd.read_csv(TESTSET_META_PATH, sep="\t")
    train_df, val_df = train_val_split(df, fold=args.fold, seed=seed)

    print("Loaded data")
    print("Train set size: {}".format(len(train_df)))
    print("Validation set size: {}".format(len(val_df)))
    print("Test set size: {}".format(len(test_df)))
    print("\nTrain")

    train(
        model=model,
        train_loader=TrainLoader(
            df=train_df,
            features_loader=ImageLoader(
                target_size=IMG_SIZE,
                augment=True,
                center_crop=False,
                norm=False,
                n_channels=None,
            ),
            batch_size=BATCH_SIZE,
            device=DEVICE,
        ),
        val_loader=TestLoader(
            df=val_df,
            features_loader=ImageLoader(
                target_size=IMG_SIZE,
                augment=False,
                center_crop=False,
                norm=False,
                n_channels=None,
            ),
            batch_size=BATCH_SIZE,
            device=DEVICE,
        ),
        val_df=val_df,
        optimizer=torch.optim.Adam(model.parameters(), lr=LR),
        criterion=Custom_Loss(temperature=args.temperature),
        num_epochs=args.n_epochs,
        model_path=CHECKPOINT_PATH,
        history_path=BASE_DIR,
        sceduler_patience=args.sceduler_patience,
        early_stopping_patience=args.early_stopping_patience,
    )

    print("\nSubmission")
    test_loader = TestLoader(
        df=test_df,
        features_loader=ImageLoader(
            target_size=IMG_SIZE,
            augment=False,
            center_crop=False,
            norm=False,
            n_channels=None,
        ),
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    model.to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    embeds = inference(model, test_loader)
    submission = get_ranked_list(embeds=embeds, top_size=500, annoy_num_trees=256)
    save_submission(submission, os.path.join(BASE_DIR, "submission_100.txt"), top=100)
    save_submission(submission, os.path.join(BASE_DIR, "submission_500.txt"), top=500)


if __name__ == "__main__":
    main()

# python3 train_main.py --base-dir='/app/_data/artist_data/' --save-dir='ny_2_1' --fold=1 --batch-size=512 --n-channels=512 --emb-len=256 --n-epochs=150 --kernel-size=3 --temperature=0.01 --sceduler-patience=4 --stopping-patience=13