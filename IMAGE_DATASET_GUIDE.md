# Week 15 Image Dataset and Evaluation Guide

## Supported ZIP structures

### Preferred predefined folders

```text
dataset.zip
├── train/
│   ├── class_1/
│   └── class_2/
├── val/                 # optional when test exists
│   ├── class_1/
│   └── class_2/
└── test/                # optional
    ├── class_1/
    └── class_2/
```

### Class folders only

```text
dataset.zip
├── class_1/
└── class_2/
```

## Evaluation priority

1. A clean supplied split is preserved.
2. A contaminated supplied split is replaced by a group-aware holdout when related sequences or exact duplicates can be identified.
3. A class-folder-only ZIP also uses group-aware splitting when repeated groups are detected.
4. A stratified random image-level holdout is the fallback when no reliable grouping information exists.

## Sequence names

The app conservatively recognizes patterns such as:

```text
paper01-000.png
paper01-005.png
paper02-000.png
```

All files from `paper01` are kept on one side. Generic names such as `image_001.png` are not automatically treated as one sequence because that would incorrectly combine many independent images.

## Metric labels

- **Validation accuracy:** reported on a supplied validation folder.
- **Final test accuracy:** reported on a separate test folder.
- **Group-aware holdout accuracy:** reported on complete groups withheld by the app.
- **Random holdout accuracy:** reported after a disclosed image-level split and interpreted cautiously.

## Classroom rule

A complex model is not proven useful by a high score alone. Students must first verify that known group overlap is zero and identify the evaluation design used.
