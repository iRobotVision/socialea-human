# MHTraj: A Multi-Domain Hybrid Graph Neural Network With Causal-Spatial Modeling for Multi-Agent Trajectory Prediction
## Getting Started

### Environment Setup


1. Set up a python environment
```
conda create -n mart python=3.8
conda activate mart
```

2. Install requirements using the following command.
```
pip install -r requirements.txt
```

## Train & Evaluation

<!-- * Trained and evaluated on NVIDIA GeForce RTX 3090 with python 3.8. -->

### NBA Dataset

* Download the [dataset](https://github.com/gist-ailab/MART/tree/main/datasets/nba) and place it in ./datasets/nba/```

* Train MART on the NBA dataset

  ```
  python main_nba.py --config ./configs/mart_nba.yaml --gpu $GPU_ID
  ```

* Test MART on the NBA dataset after training
  ```
  python main_nba.py --config ./configs/mart_nba.yaml --gpu $GPU_ID --test
  ```

### ETH-UCY Dataset
* The dataset is included in ```./datasets/ethucy/```
* Train MART on the ETH-UCY dataset
  ```
  chmod +x ./scripts/train_eth_all.sh
  ./scripts/train_eth_all.sh ./configs/mart_eth.yaml $GPU_ID
  ```

* Test MART on the ETH-UCY dataset after training
  ```
  chmod +x ./scripts/test_eth_all.sh
  ./scripts/test_eth_all.sh ./configs/mart_eth.yaml $GPU_ID
  ```

### SDD Dataset
* The dataset is included in ```./datasets/stanford/```

* Train MART on the SDD dataset

  ```
  python main_sdd.py --config ./configs/mart_sdd.yaml --gpu $GPU_ID
  ```

* Test MART on the SDD dataset after training
  ```
  python main_sdd.py --config ./configs/mart_sdd.yaml --gpu $GPU_ID --test
  ```

## Main Results
### NBA Dataset
*
  ```
  minADE (4.0s): 0.699
  minFDE (4.0s): 0.881
  ```


### ETH-UCY Dataset
```
minADE Table
       ETH    HOTEL    UNIV    ZARA1    ZARA2    AVG
       0.35    0.14    0.25    0.17    0.13    0.21    

minFDE Table
       ETH    HOTEL    UNIV    ZARA1    ZARA2    AVG
       0.47    0.22    0.45    0.29    0.22    0.33    
```

### SDD Dataset
```
minADE: 7.43
minFDE: 11.82
```

## How to reproduce results

### NBA Dataset

* The checkpoint is included in ```./checkpoints/mart_nba_reproduce/```

  ```
  python main_nba.py --config ./configs/mart_nba_reproduce.yaml --gpu $GPU_ID --test
  ```
* The results will be saved in ```./results/nba_result.csv```

### ETH-UCY Dataset

* The checkpoints are included in ```./checkpoints/mart_eth_reproduce/```
  ```
  ./scripts/test_eth_all.sh ./configs/mart_eth_reproduce.yaml $GPU_ID
  ```
* The results will be saved in ```./results/$SUBSET-NAME_result.csv```

### SDD Dataset

* The checkpoint is included in ```./checkpoints/mart_sdd_reproduce/```
  ```
  python main_sdd.py --config ./configs/mart_sdd_reproduce.yaml --gpu $GPU_ID --test
  ```
* The results will be saved in ```./results/sdd_result.csv```

## 🤗 Acknowledgement
* Our work benefits from [MART](https://github.com/gist-ailab/MART) in the training and testing pipeline—special thanks to the authors for their great contribution!
