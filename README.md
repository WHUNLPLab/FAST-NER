# Dilated Convolutional Networks Incorporating Soft Entity Type Constraints for Distant Supervised Relation Extraction

Source code for IJCNN 2019 paper: [Dilated Convolutional Networks Incorporating Soft Entity Type Constraints for Distant Supervised Relation Extraction]()

## Overview

![](figs/Architecture.png)

The architecture of our model, where $m_i$ indicates the original sentence for an entity pair. $p_j$ indicates the sentence feature encoded by DCNNs. Before calculating attention weight $\alpha_i$ for mention mi, its entity types are dynamically ajusted via the model prediction and current type labels to denoise and acquire soft entity type. Taking soft entity types into account, $\alpha_i$ is specified to the entity types of $m_i$. Finally, weighted summation is applied on all sentence features for producing bag feature $\mathbf{b}$.

## Dependencies

* Pytorch 1.0.1
* tqdm 4.31.1
* scikit-learn 0.20.3
* Compatible with Python 3.X

## Dataset

*  Like the paper "Neural Relation Extraction with Selective Attention over Instances", we use [Riedel NYT dataset](http://iesl.cs.umass.edu/riedel/ecml/) for evaluation.
*  Our dataset can be downloaded from [here]().

## Training from scratch

* Options for preprocessing, traning and testing
  
  All options can be modified in `nre/options.py`

* Preprocessing the dataset:
  ```
  python preprocess.py
  ```

* Train the model:
  ```
  python train.py -batch_size 160 -model_type PDCNN+TATT -gpu_num 1
  ```
  After training, the trained model can be found in `ckpt` directory.

* Test the model:
  ```
  python test.py -batch_size 160 -model_type PDCNN+TATT -gpu_num 1 -pretrain_model ckpt/model_step_best.pt
  ```

## Evaluation Results

![](figs/result.png)

## Citation:

Please cite the following paper if you use this code in your work.

```
None
```

For any clarification, please create an issue or contact huweilong@whu.edu.cn.