# ViT_experiment

notebook/下でコマンド CUDA_LAUNCH_BLOCKING=1 python3 ViT_trainer.py を実行することで学習することができる

.gitignoreによって
inputs/
trash/
trained_model/
が見えなくなっているのでcloneの際は作成が必要

inputs/におけるcifar10/などのディレクトリはデータセットに合わせて適宜作成が必要


$ tree
.
├── data
│   ├── inputs
│   │   ├── cifar10
│   │   │   └── .keep
│   │   └── cifar100
│   │       └── .keep
│   ├── outputs
│   │   ├── history
│   │   │   └── .keep
│   │   └── trained_model
│   │       └── .keep
│   └── trush
│       └── .keep
├── features
│   └── .keep
├── modules
│   └── .keep
├── notebook
│   └── .keep
└── settings
    └── .keep
