# RandomResidualGCN

## How to Use

```python
python main.py -pre_train_model ours
```

## Experiments

- `embedding_dim = 8`
- `epoch_size = 64`

### umls

||Hit@10|MR|MRR|
|:-:|:-:|:-:|:-:|
|No Pre-train|56.9%|11.60|0.208|
|Baseline|57.6%|11.46|0.213|
|**Ours**|59.9%|11.05|0.218|

### kinship

||Hit@10|MR|MRR|
|:-:|:-:|:-:|:-:|
|No Pre-train|71.2%|8.73|0.241|
|Baseline|72.7%|8.45|0.245|
|**Ours**|73.6%|8.20|0.255|
