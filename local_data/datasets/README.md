# Datasets

All experiments use the [fastMRI](https://fastmri.org/) single-coil knee and multi-coil brain data
(train and validation splits only). Request access on the fastMRI website, download the four splits
below, and extract the knee archives into `fastMRI/knee/` and the brain archives into `fastMRI/brain/`.
Please cite the fastMRI papers listed on the website when you use the data.

| Data | Split directory | Files | Slices |
|---|---|---|---|
| knee single-coil train | `knee/singlecoil_train` | 973 | 34,742 |
| knee single-coil val | `knee/singlecoil_val` | 199 | 7,135 |
| brain multi-coil train | `brain/multicoil_train` | 4,469 | 70,748 |
| brain multi-coil val | `brain/multicoil_val` | 1,378 | 21,842 |

## Expected layout

The data is read from `PATH_DATASETS` (default `./local_data/datasets/fastMRI/`, see
[`../constants.py`](../constants.py)). A split directory is `<data_path>/<anatomy>/<challenge>_<split>`:

```
local_data/datasets/
├── fastMRI/
│   ├── knee/{singlecoil_train,singlecoil_val}/file1000001.h5, ...
│   └── brain/{multicoil_train,multicoil_val}/file_brain_AXFLAIR_200_6002425.h5, ...
```

If the data lives elsewhere, link it (`ln -s /path/to/fastMRI local_data/datasets/fastMRI`), edit
`PATH_DATASETS`, or pass `--data_path /path/to/fastMRI` to `train.py` and `evaluate.py`.
