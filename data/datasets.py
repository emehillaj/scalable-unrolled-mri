"""fastMRI slice datasets (``<data_path>/<anatomy>/<challenge>_<split>``) and masks."""

import hashlib
import os
import pickle
import socket
import warnings
from pathlib import Path

from fastmri.data import SliceDataset
from fastmri.data.subsample import RandomMaskFunc

from local_data.constants import PATH_CACHE, PATH_DATASETS
from training.distributed import print0
from utils.misc import CENTER_FRACTIONS

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_mask_func(cfg: dict) -> RandomMaskFunc:
    """Random column mask; the center fraction defaults to the fastMRI value for the acceleration."""
    acceleration = cfg["acceleration"]
    center_fraction = cfg.get("center_fraction")
    if center_fraction is None:
        if acceleration not in CENTER_FRACTIONS:
            raise ValueError(
                f"no default center fraction for acceleration {acceleration}; "
                "set center_fraction explicitly"
            )
        center_fraction = CENTER_FRACTIONS[acceleration]
    return RandomMaskFunc([center_fraction], [acceleration])


def _repo_path(path) -> Path:
    """Resolve a relative path from the repo root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def read_file_list(path) -> list:
    """Sorted unique file basenames from a text file (one per line, '#' comments)."""
    names = set()
    with open(_repo_path(path)) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                names.add(Path(line).name)
    if not names:
        raise ValueError(f"file list {path} is empty")
    return sorted(names)


class _FileNameFilter:
    """raw_sample_filter keeping the slices of the listed files (picklable)."""

    def __init__(self, names):
        self.names = frozenset(names)

    def __call__(self, raw_sample) -> bool:
        return raw_sample.fname.name in self.names


def _read_cache(cache_file: Path) -> dict:
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except Exception as e:  # an unreadable cache is rebuilt, never fatal
        warnings.warn(f"ignoring unreadable dataset cache {cache_file}: {e}")
        return {}


def _listing(root: Path) -> list:
    """Entry names of a split directory, to notice added or removed files."""
    return sorted(p.name for p in root.iterdir())


def _slice_dataset(root: Path, challenge: str, transform, cache_file: Path, raw_sample_filter):
    kwargs = dict(
        root=root,
        challenge=challenge,
        transform=transform,
        use_dataset_cache=True,
        raw_sample_filter=raw_sample_filter,
    )
    listing = _listing(root)
    cache = _read_cache(cache_file)
    if root in cache:
        stored = cache.get((root, "listing"))
        if stored == listing:
            return SliceDataset(dataset_cache_file=cache_file, **kwargs)
        if stored is not None:
            warnings.warn(f"the files in {root} changed since it was indexed; indexing it again")

    # SliceDataset writes the cache in place; write a private file and os.replace it, so that
    # ranks indexing at the same time never read a partial pickle.
    print0(f"indexing {root}: reading every file header (slow for brain on network storage), "
           f"cached in {cache_file}")
    tag = f"{socket.gethostname()}.{os.getpid()}"
    scan_file = cache_file.with_name(f"{cache_file.name}.{tag}.scan")
    tmp_file = cache_file.with_name(f"{cache_file.name}.{tag}.tmp")
    try:
        dataset = SliceDataset(dataset_cache_file=scan_file, **kwargs)
        cache = _read_cache(cache_file)
        cache[root] = dataset.raw_samples
        cache[(root, "listing")] = listing
        with open(tmp_file, "wb") as f:
            pickle.dump(cache, f)
        os.replace(tmp_file, cache_file)
    finally:
        scan_file.unlink(missing_ok=True)
        tmp_file.unlink(missing_ok=True)
    return dataset


def build_dataset(cfg: dict, split: str, transform, data_path=None) -> SliceDataset:
    """fastMRI SliceDataset for "train" or "val", indexed in PATH_CACHE.

    data_path defaults to cfg["data_path"], then PATH_DATASETS. For the train split,
    cfg["train_file_list"] restricts the files to the listed basenames.
    """
    anatomy, challenge = cfg["anatomy"], cfg["challenge"]
    if data_path is None:
        data_path = cfg.get("data_path")
    data_path = Path(data_path).expanduser() if data_path else _repo_path(PATH_DATASETS)
    root = (data_path / anatomy / f"{challenge}_{split}").resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"fastMRI split directory not found: {root}. Pass --data_path or edit "
            "PATH_DATASETS in local_data/constants.py (see local_data/datasets/README.md)."
        )

    cache_name = f"{anatomy}_{challenge}_{split}"
    names, raw_sample_filter = None, None
    list_path = cfg.get("train_file_list") if split == "train" else None
    if list_path:
        names = read_file_list(list_path)
        digest = hashlib.sha1("\n".join(names).encode()).hexdigest()[:8]
        cache_name += f"_list-{digest}"
        raw_sample_filter = _FileNameFilter(names)
    cache_dir = _repo_path(PATH_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = _slice_dataset(root, challenge, transform, cache_dir / f"{cache_name}.pkl", raw_sample_filter)

    if names is not None:
        found = {sample.fname.name for sample in dataset.raw_samples}
        missing = sorted(set(names) - found)
        if missing:
            warnings.warn(f"{len(missing)} of {len(names)} files in {list_path} are not in {root}, "
                          f"e.g. {missing[:5]}")
    if len(dataset) == 0:
        raise ValueError(f"no slices found in {root}")
    return dataset
