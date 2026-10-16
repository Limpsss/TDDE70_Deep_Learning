# TDDE70 – CLIP Fine-tuning on RSICD
Project work for the course TDDE70 Deep Learning at Linköping University (spring 2026).

This project fine-tunes OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) on remote-sensing image–caption
pairs from RSICD, then compares zero-shot classification accuracy of the pretrained
and fine-tuned model on the test split (54.25% → 77.13%).

## Layout

```
data/                 RSICD dataset (not in git)
  RSICD_images/         10 921 images
  txtclasses_rsicd/     one .txt per class, listing its image filenames
  dataset_rsicd.json    captions (5 per image) + train/val/test split
src/
  dataset.py            RSICD torch Dataset (splits A/B/C, caption modes)
  utils.py              project paths, class list, checkpoint loader
  train.py              fine-tunes CLIP on split B -> checkpoints/, results/
  evaluate_both.py      zero-shot eval of pretrained + fine-tuned on split C
  plot_confusion_matrix.py  renders results/finetuned_crosstab.csv as a PDF
checkpoints/          finetuned.pt (best epoch by validation loss)
results/              loss curve, crosstabs (confusion matrices), plots
```

## Setup

Download the RSICD dataset from <https://github.com/201528014227051/RSICD_optimal>
(the README there links Google Drive / Baidu mirrors) and place `RSICD_images/`,
`txtclasses_rsicd/` and `dataset_rsicd.json` in `data/`.

```sh
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running

Scripts resolve paths relative to the repo, so they can be run from anywhere:

```sh
source .venv/bin/activate      # or .venv/bin/activate.fish
python src/train.py            # ~10 epochs, writes checkpoints/finetuned.pt
python src/evaluate_both.py    # writes results/*_crosstab.csv
python src/plot_confusion_matrix.py
```
