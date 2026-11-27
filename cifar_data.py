# cifar_data.py - data code for lab assignment 1
# everyone imports from here so we all use the same split, stats and seeds.
# based on the lab 2 part A code so train_epoch / evaluate from there still work
#
# how to use:
#   data = load_cifar10()
#   split = get_split(data["train"])
#   train_ids, val_ids, stats = split["train_ids"], split["val_ids"], split["stats"]
#   loaders = make_loaders(run, data["train"], train_ids, val_ids)
#
# don't use the test set until the final evaluation
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as TF

# don't change these without telling everyone - the saved split depends on them
DATA_ROOT = Path("data")
SPLIT_FILE = Path("splits/cifar10_split.json")
SPLIT_SEED = 2223    # same as part A
VAL_FRACTION = 0.1   # 45000 train / 5000 val

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# both A9 models use 32x32 so no resizing for now
INPUT_SIZES = {"small_cnn": 32, "alexnet": 32, "resnet18": 32}

MIRROR_URL = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"


def set_seed(seed, deterministic=True):
    # deterministic cudnn is a bit slower, and results can still be slightly different on another GPU
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    # only matters if num_workers > 0
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_cifar10(root=DATA_ROOT):
    root = Path(root)
    try:
        train = datasets.CIFAR10(root, train=True, download=False)
    except RuntimeError:
        # not downloaded yet - try the part A mirror first, then the official site
        try:
            from torchvision.datasets.utils import download_and_extract_archive
            download_and_extract_archive(MIRROR_URL, download_root=str(root),
                                         filename="cifar-10-python.tar.gz", md5=ARCHIVE_MD5)
        except Exception as e:
            warnings.warn(f"mirror didn't work ({e}), trying the official site")
            datasets.CIFAR10(root, train=True, download=True)
        train = datasets.CIFAR10(root, train=True, download=False)
    test = datasets.CIFAR10(root, train=False, download=False)
    assert list(train.classes) == CLASS_NAMES
    return {"train": train, "test": test, "class_names": CLASS_NAMES}


def make_split(labels, seed=SPLIT_SEED, val_fraction=VAL_FRACTION):
    # image id = index in the official training set (0-49999)
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels)
    train_ids, val_ids = train_test_split(np.arange(len(labels)), test_size=val_fraction,
                                          stratify=labels, random_state=seed)
    return np.sort(train_ids), np.sort(val_ids)


def compute_stats(images, ids):
    # only use the training ids here, otherwise info from val/test leaks in
    ids = np.asarray(ids)
    total = np.zeros(3)
    total_sq = np.zeros(3)
    n_pixels = 0
    for start in range(0, len(ids), 1024):
        pixels = images[ids[start:start + 1024]].astype(np.float64) / 255
        total += pixels.sum(axis=(0, 1, 2))
        total_sq += np.square(pixels).sum(axis=(0, 1, 2))
        n_pixels += pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
    mean = total / n_pixels
    std = np.sqrt(np.maximum(total_sq / n_pixels - mean**2, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist()}


def get_split(train_dataset, path=SPLIT_FILE, seed=SPLIT_SEED, val_fraction=VAL_FRACTION):
    # loads the saved split, or makes it and saves it on the first run.
    # the json is in git so we all train on the same images
    path = Path(path)
    labels = np.asarray(train_dataset.targets)
    new_train, new_val = make_split(labels, seed, val_fraction)

    if path.exists():
        saved = json.loads(path.read_text())
        train_ids = np.array(saved["train_ids"])
        val_ids = np.array(saved["val_ids"])
        stats = saved["stats"]
        if not np.array_equal(train_ids, new_train):
            # can happen with a different sklearn version - the saved file wins
            warnings.warn("split from the seed doesn't match the saved one, using the saved file")
    else:
        train_ids, val_ids = new_train, new_val
        stats = compute_stats(train_dataset.data, train_ids)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "seed": seed,
            "val_fraction": val_fraction,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "stats": stats,
            "train_ids": train_ids.tolist(),
            "val_ids": val_ids.tolist(),
        }))

    assert len(set(train_ids.tolist()) & set(val_ids.tolist())) == 0, "train and val overlap"
    assert len(train_ids) + len(val_ids) == len(labels)
    return {"train_ids": train_ids, "val_ids": val_ids, "stats": stats,
            "seed": seed, "val_fraction": val_fraction}


def training_augmentation(padding=4, crop=True, flip=True):
    # for task 4 - random crop + horizontal flip.
    # no vertical flip since upside down horses/ships etc don't really make sense
    steps = []
    if crop: steps.append(T.RandomCrop(32, padding=padding, padding_mode="reflect"))
    if flip: steps.append(T.RandomHorizontalFlip(p=0.5))
    return T.Compose(steps)


class RawImages(Dataset):
    # gives (image in [0,1], label, image id) like in part A.
    # normalising is done later in prepare_inputs (needed for the task 5 attack)

    def __init__(self, dataset, ids, augment=None):
        self.dataset = dataset
        self.ids = [int(i) for i in ids]
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image, target = self.dataset[image_id]
        raw = TF.to_tensor(image)
        if self.augment is not None:
            raw = self.augment(raw)
        return raw, int(target), image_id


def prepare_inputs(raw, stats, size=32):
    if raw.shape[-1] != size or raw.shape[-2] != size:
        raw = nn.functional.interpolate(raw, size=(size, size), mode="bilinear",
                                        align_corners=False, antialias=True)
    mean = raw.new_tensor(stats["mean"])[None, :, None, None]
    std = raw.new_tensor(stats["std"])[None, :, None, None]
    return (raw - mean) / std


def make_loaders(run, train_dataset, training_ids, validation_ids, num_workers=0):
    # works with the Run dataclass from part A. only "train" gets augmented (if run.augmentation is on)
    augment = training_augmentation() if getattr(run, "augmentation", False) else None
    settings = dict(batch_size=run.batch_size, num_workers=num_workers,
                    worker_init_fn=seed_worker if num_workers > 0 else None,
                    pin_memory=torch.cuda.is_available())
    return {
        "train": DataLoader(RawImages(train_dataset, training_ids, augment), shuffle=True,
                            generator=torch.Generator().manual_seed(run.seed), **settings),
        "train_eval": DataLoader(RawImages(train_dataset, training_ids), **settings),
        "val": DataLoader(RawImages(train_dataset, validation_ids), **settings),
    }


def make_test_loader(test_dataset, batch_size=256, num_workers=0):
    # final evaluation only!
    ids = np.arange(len(test_dataset))
    return DataLoader(RawImages(test_dataset, ids), batch_size=batch_size,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
