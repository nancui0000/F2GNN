# F2GNN

Codes for ICDM 2023 paper: *Equipping Federated Graph Neural Networks with Structure-aware Group Fairness*

## Description

A detailed explanation of the project, including its purpose, usage, and functionality. Explain the problems it solves, its applications, and how it works.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

For the pokec-z dataset:

```bash
python main.py --alpha=1e-06 --dataset='pokec-z' --dropout=0.5 --ego_number=30 --gpu=0 --lambda1=0.5 --local_ep=20 --lr=0.0001 --num_hidden=64 --num_hops=3 --seed=31 --tau=4 --tau_combine=0.01 --weight_decay=0.001
```

For the pokec-n dataset:

``````bash
python main.py  --alpha=1e-06 --dataset='pokec-n' --dropout=0.1 --ego_number=30 --gpu=0 --lambda1=8.0 --local_ep=15 --lr=0.0001 --num_hidden=64 --num_hops=3 --seed=47 --tau=4 --tau_combine=0.001 --weight_decay=0.0001
``````



## Results

For the pokec-z dataset:

![image-20230927115801292](C:\Users\cuina\AppData\Roaming\Typora\typora-user-images\image-20230927115801292.png)

For the pokec-n dataset:

![image-20230927113620536](C:\Users\cuina\AppData\Roaming\Typora\typora-user-images\image-20230927113620536.png)

