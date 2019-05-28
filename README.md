# MSADnn
Mammography Screening Abnormality Detection with neural networks.
Michal Kowalski's master project for
Molecular Biotechnology Msc programme at Jagiellonian University.


All of models were trained on DDSM dataset, prepared with [ddsm_tools](https://github.com/fjeg/ddsm_tools) mentioned in [CBIS-DDSM publication](https://www.nature.com/articles/sdata2017177) and own code posted as [dataset_utils]().
## Dataset preparation utils

## Keras(Tensofrlow backend) Tiny YOLO abnormality detector code
Whole architecture is based on [qqweee's](https://github.com/qqwweee/keras-yolo3) implementation with slight changes
## Weights for models
There are six Tiny-YOLO models trained for (two models for one task)
Input shapes: 3424x2432px and 4480x4480px
Tasks: Detection of three classes (normal, calcification, mass), detection of two classes (mass, calcification), detection of abnormalities.
https://drive.google.com/drive/folders/1NAcFPtR1GzsAEcDaVssot8yWzMGjIkfS?usp=sharing
