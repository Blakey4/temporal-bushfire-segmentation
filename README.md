**Temporal Bushfire Segmentation**
UTS 310005 Machine Learning project to investigate the performance difference between uni and bi temporal burnt area segmentation. 

The primary questions is: Does bi-temporal imagery significantly improve burnt-area segmentation compared with post-fire-only imagery under otherwise matched conditions?

The secondary focus depending on whether time allows the building of an Australian dataset is to test generalisaiblty of these models trained on Greek bushfires then tested on unseen Australian bushfires

---
## Repo Structure
Initial planned repo structure. individual files in each subdirectory may change.
```
bushfire-ml/
│
├── README.md
├── requirements.txt
├── configs/
│   └── baseline.yaml                 # optional
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # load image/mask pairs
│   ├── models.py                     # U-Net / model construction
│   ├── losses.py                     # BCE, Dice, BCE+Dice etc.
│   ├── metrics.py                    # IoU, Dice, precision, recall
│   ├── train.py                      # training + validation loops
│   ├── evaluate.py                   # event-level evaluation
│   └── visualisation.py              # predictions / error overlays
│
├── notebooks/
│   └── assignment_demo.ipynb         # THE public Colab notebook
│
├── data/
│   └── README.md                     # explain where data comes from
│
├── results/
│   ├── metrics.csv
│   ├── figures/
│   └── predictions/
│
├── PROJECT_PLAN.md
└── JOURNAL.md
```
