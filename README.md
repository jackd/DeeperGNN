# Simplified DAGNN

This is a slightly modified version of the [official pytorch DAGNN implementation](https://github.com/mengliu1998/DeeperGNN) to demonstrate the effectiveness of simpler models with similar performance. We replace the deep adaptive weighted propagation with a single static propagation, giving __simple static__ DAGNN (SS-DAGNN). This is documented in [It's PageRank All The Way Down: Simplifying Deep Graph Networks (SDM23)](https://github.com/jackd/ppr-gnn-sdm23).

## Requirements

* PyTorch
* PyTorch Geometric
* NetworkX
* tdqm

## Results

| Dataset    | DAGNN        | SS-DAGNN     |
|------------|--------------|--------------|
| Cora       | 84.15 ± 0.56 | 84.32 ± 0.64 |
| Citeseer   | 73.18 ± 0.50 | 73.08 ± 0.51 |
| PubMed     | 80.62 ± 0.49 | 80.59 ± 0.47 |
| CS         | 92.63 ± 0.53 | 92.82 ± 0.34 |
| Physics    | 94.03 ± 0.41 | 93.83 ± 1.12 |
| Computer   | 83.70 ± 1.45 | 83/49 ± 1.12 |
| Photo      | 91.32 ± 1.31 | 91.26 ± 1.37 |
| ogbn-arxiv | 72.01 ± 0.26 | 71.41 ± 0.24 |

```bash
python dagnn.py --dataset=Cora --weight_decay=0.005 --K=10 --dropout=0.8 --runs=10
# Val Loss: 0.6255, Test Accuracy: 0.8415 ± 0.0056, Duration: 4.113
python dagnn.py --dataset=Cora --weight_decay=0.005 --K=10 --dropout=0.8 --runs=10 --static
# Val Loss: 0.6185, Test Accuracy: 0.8432 ± 0.0064, Duration: 4.942

python dagnn.py --dataset=CiteSeer --weight_decay=0.02 --K=10 --dropout=0.5 --runs=10
# Val Loss: 1.1492, Test Accuracy: 0.7318 ± 0.0050, Duration: 6.108
python dagnn.py --dataset=CiteSeer --weight_decay=0.02 --K=10 --dropout=0.5 --runs=10 --static
# Val Loss: 1.1361, Test Accuracy: 0.7308 ± 0.0051, Duration: 5.092

python dagnn.py --dataset=PubMed --weight_decay=0.005 --K=20 --dropout=0.8 --runs=10
# Val Loss: 0.4879, Test Accuracy: 0.8062 ± 0.0049, Duration: 14.491
python dagnn.py --dataset=PubMed --weight_decay=0.005 --K=20 --dropout=0.8 --runs=10 --static
# Val Loss: 0.4899, Test Accuracy: 0.8059 ± 0.0047, Duration: 8.539


python dagnn.py --dataset=cs --weight_decay=0 --K=5 --dropout=0.8 --runs=10
# Val Loss: 0.2683, Test Accuracy: 0.9263 ± 0.0053, Duration: 37.999
python dagnn.py --dataset=cs --weight_decay=0 --K=5 --dropout=0.8 --runs=10 --static
# Val Loss: 0.2622, Test Accuracy: 0.9282 ± 0.0034, Duration: 35.065

python dagnn.py --dataset=physics --weight_decay=0 --K=5 --dropout=0.8 --runs=10
# Val Loss: 0.2054, Test Accuracy: 0.9403 ± 0.0041, Duration: 21.109
python dagnn.py --dataset=physics --weight_decay=0 --K=5 --dropout=0.8 --runs=10 --static
# Val Loss: 0.1929, Test Accuracy: 0.9383 ± 0.0112, Duration: 19.484

python dagnn.py --dataset=computers --weight_decay=0.00005 --K=5 --dropout=0.5 --epochs=3000 --early_stopping=300 --runs=10
# Val Loss: 0.5484, Test Accuracy: 0.8370 ± 0.0145, Duration: 27.177
python dagnn.py --dataset=computers --weight_decay=0.00005 --K=5 --dropout=0.5 --epochs=3000 --early_stopping=300 --runs=10 --static
# Val Loss: 0.5844, Test Accuracy: 0.8349 ± 0.0112, Duration: 25.012

python dagnn.py --dataset=photo --weight_decay=0.0005 --K=5 --dropout=0.5 --runs=10
# Val Loss: 0.3532, Test Accuracy: 0.9132 ± 0.0131, Duration: 10.848
python dagnn.py --dataset=photo --weight_decay=0.0005 --K=5 --dropout=0.5 --runs=10 --static
# Val Loss: 0.3731, Test Accuracy: 0.9126 ± 0.0137, Duration: 8.899

python main_ogbnarxiv.py
# All runs:
# Highest Train: 80.40 ± 0.14
# Highest Valid: 73.04 ± 0.08
#   Final Train: 78.26 ± 0.74
#    Final Test: 72.01 ± 0.26
python main_ogbnarxiv.py --static
# All runs:
# Highest Train: 79.35 ± 0.18
# Highest Valid: 72.50 ± 0.09
#   Final Train: 78.43 ± 0.91
#    Final Test: 71.41 ± 0.24
```

To regenerate Figure 3 from the SDM23 paper,

```bash
python dagnn.py --dataset=PubMed --weight_decay=0.005 --K=20 --dropout=0.8 --runs=1 --plot
```
