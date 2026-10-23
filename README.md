# Retinal Disease Classification from OCT Scans

Transfer-learning classifier that reads an optical coherence tomography (OCT) scan and labels it **CNV**, **DME**, **DRUSEN** or **NORMAL** — the four categories ophthalmologists triage on.

**Result:** 96.3% test accuracy. Per-class F1: CNV 0.95 · DME 0.98 · DRUSEN 0.95 · NORMAL 0.98.
F1 is tracked per class because DRUSEN is the smallest and most-confused class; a model can hit high overall accuracy while quietly failing it.

## Data

Kermany et al. 2018 ([Cell](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)): 84,495 OCT images, graded by a tiered panel of retinal specialists, split into train / val / test. Available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018).

## Method

- ImageNet-pretrained convolutional backbone, top layers replaced with a 4-way softmax head; base frozen for the first epochs, then fine-tuned end to end.
- Augmentation: random flips, small rotations, brightness jitter (OCT scans vary a lot in contrast between machines).
- Class-weighted loss to compensate for the DRUSEN / NORMAL imbalance.
- Trained 30 epochs; `tensorflow_addons` F1 metric logged per class each epoch.

## Run

Open `RETINA-DAMAGE-CLASSIFICATION.ipynb`, point the data path at the Kermany2018 folder, run all cells. TensorFlow 2.x, tensorflow_addons.

## What I'd do next

Grad-CAM overlays so the prediction can be checked against the lesion the model is looking at; test-time augmentation; export to TFLite for a clinic-side demo.
