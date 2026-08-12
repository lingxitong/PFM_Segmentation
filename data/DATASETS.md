# Datasets: Download & Preprocessing

Download links and preprocessing scripts for the pathology segmentation datasets used in this repository.

After downloading the raw data, run the corresponding script under [`preprocess_datasets/`](./preprocess_datasets/) to produce a unified `images/` + `masks/` layout, then build a dataset JSON as described in the main README (see [`example.json`](./example.json)).

## Workflow

1. Download each dataset from the link in the table below.
2. Edit the input/output paths in the matching preprocess script (placeholders default to `/path/to/PFM_Segmentation_Data/...`).
3. Run the script from the repository root, for example:

```bash
python data/preprocess_datasets/CPM15.py
python data/preprocess_datasets/GlaS.py
# ... same pattern for other datasets
```

4. Write processed `image_path` / `mask_path` entries into a JSON file and set `dataset.json_file` in `configs/config.yaml`.

## Download & preprocess scripts

| Dataset | Download | Preprocess script |
|---------|----------|-------------------|
| BCSS | [GitHub](https://github.com/PathologyDataScience/BCSS) | [`BCSS.py`](./preprocess_datasets/BCSS.py) (usually no extra conversion) |
| CoCaHis | [cocahis.irb.hr](https://cocahis.irb.hr/) | [`CoCaHis.py`](./preprocess_datasets/CoCaHis.py) |
| CONIC2022 | [Grand Challenge](https://conic-challenge.grand-challenge.org/) | [`CoNIC2022.py`](./preprocess_datasets/CoNIC2022.py) |
| CoNSeP | [OpenDataLab](https://opendatalab.com/OpenDataLab/CoNSeP/tree/main) | [`CoNSeP.py`](./preprocess_datasets/CoNSeP.py) |
| COSAS24 | [Grand Challenge](https://cosas.grand-challenge.org/) | [`COSAS24.py`](./preprocess_datasets/COSAS24.py) |
| CPM15 | [Google Drive](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK) | [`CPM15.py`](./preprocess_datasets/CPM15.py) |
| CPM17 | [Google Drive](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK) | [`CPM17.py`](./preprocess_datasets/CPM17.py) |
| CRAG | [Warwick TIA](https://warwick.ac.uk/fac/cross_fac/tia/data/mildnet/) | [`CRAG.py`](./preprocess_datasets/CRAG.py) |
| EBHI | [Kaggle](https://www.kaggle.com/datasets/alibabaei78/ebhi-seg) | [`EBHI.py`](./preprocess_datasets/EBHI.py) |
| GlaS | [DatasetNinja](https://datasetninja.com/gland-segmentation) | [`GlaS.py`](./preprocess_datasets/GlaS.py) |
| Janowczyk | [andrewjanowczyk.com](https://andrewjanowczyk.com/use-case-1-nuclei-segmentation/) | [`Janowczyk.py`](./preprocess_datasets/Janowczyk.py) |
| MoNuSeg (Kumar) | [Google Drive](https://drive.google.com/drive/folders/1bI3RyshWej9c4YoRW-q7lh7FOFDFUrJ) | [`Kumar.py`](./preprocess_datasets/Kumar.py) |
| Lizard | [Kaggle](https://www.kaggle.com/datasets/aadimator/lizard-dataset) | [`Lizard.py`](./preprocess_datasets/Lizard.py) |
| NuCLS | [Grand Challenge](https://nucls.grand-challenge.org/) | [`NuCLS.py`](./preprocess_datasets/NuCLS.py) |
| PanNuke | [Warwick TIA](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke) | [`PanNuke.py`](./preprocess_datasets/PanNuke.py) |
| RINGS | [Mendeley](https://data.mendeley.com/datasets/h8bdwrtnr5/1) | [`RINGS.py`](./preprocess_datasets/RINGS.py) |
| TNBC | [peterjacknaylor.github.io](https://peterjacknaylor.github.io/data/) | [`TNBC.py`](./preprocess_datasets/TNBC.py) |
| WSSS4LUAD | [Grand Challenge](https://wsss4luad.grand-challenge.org/) | [`WSSS4LUAD.py`](./preprocess_datasets/WSSS4LUAD.py) |
