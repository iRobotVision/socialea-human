# MHTraj: A Multi-Domain Hybrid Graph Neural Network With Causal-Spatial Modeling for Multi-Agent Trajectory Prediction
## Getting Started

### Environment Setup


1. Set up a python environment
```
conda create -n mhtraj python=3.8
conda activate mhtraj
```

2. Install requirements using the following command.
```
pip install -r requirements.txt
```

## Train & Evaluation

<!-- * Trained and evaluated on NVIDIA GeForce RTX 3090 with python 3.8. -->

### NBA Dataset

* Download the [dataset](https://github.com/gist-ailab/MART/tree/main/datasets/nba) and place it in ./datasets/nba/```

* Train MHTraj on the NBA dataset

  ```
  python main_nba.py --config ./configs/mhtraj_nba.yaml --gpu $GPU_ID
  ```

* Test MHTraj on the NBA dataset after training
  ```
  python main_nba.py --config ./configs/mhtraj_nba.yaml --gpu $GPU_ID --test
  ```

### ETH-UCY Dataset
* The dataset is included in ```./datasets/ethucy/```
* Train MHTraj on the ETH-UCY dataset
  ```
  chmod +x ./scripts/train_eth_all.sh
  ./scripts/train_eth_all.sh ./configs/mhtraj_eth.yaml $GPU_ID
  ```

* Test MHTraj on the ETH-UCY dataset after training
  ```
  chmod +x ./scripts/test_eth_all.sh
  ./scripts/test_eth_all.sh ./configs/mhtraj_eth.yaml $GPU_ID
  ```

### SDD Dataset
* The dataset is included in ```./datasets/stanford/```

* Train MHTraj on the SDD dataset

  ```
  python main_sdd.py --config ./configs/mhtraj_sdd.yaml --gpu $GPU_ID
  ```

* Test MHTraj on the SDD dataset after training
  ```
  python main_sdd.py --config ./configs/mhtraj_sdd.yaml --gpu $GPU_ID --test
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
       0.37    0.16    0.25    0.16    0.13    0.21    

minFDE Table
       ETH    HOTEL    UNIV    ZARA1    ZARA2    AVG
       0.50    0.27    0.45    0.28    0.21    0.34    
```

### SDD Dataset
```
minADE: 7.82
minFDE: 12.48
```


## 🤗 Acknowledgement
* Our work benefits from [MART](https://github.com/gist-ailab/MART) in the training and testing pipeline—special thanks to the authors for their great contribution!
