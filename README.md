# IM-Net_reproduction
This repository is the project for course Deep Generation Model. In this project, I try to reproduce [IM-Net](https://www.sfu.ca/~zhiqinc/imgan/Readme.html) in pytorch and optimize the sampling and training process.

There are two sampling methods in this repo,  `baseline` is multi resolution training and `soft_boundary` is soft boundary training process. 



## Dependencies

Install python package dependencies through pip:


```bash
pip install -r requirements.txt
```



## Dataset

We use dataset in [PartNet](https://cs.stanford.edu/~kaichun/partnet/) and voxelize the shape in $64^3$ resolution. To download our training data, please visit this [link](https://disk.pku.edu.cn:443/link/BB3144D411E61092DECED4C7F0C8ED11) (the voxelization is done by [Rundi Wu](https://github.com/ChrisWu1997/PQ-NET)) and unzip it under `data` folder.



## Sampling

Before training the shape, you need to do the data sampling by

```python
python data_processing.py
```

the sampling method in `baseline` will generate a new folder `processed_data` in the root path and another in `soft_boundary` will generate `soft_data`



## Training

After sampling the shape, simply run

```python
python main.py
CUDA_VISIBLE_DEVICES=0 python main.py # GPU user
```

Network structure

![](https://github.com/jianghd1996/IM-Net_reproduction/blob/master/result/network.JPG)



## Testing and visualization

We define the `test()` and `visualization(number)` function in the `main.py`, to use them, uncomment the following code in main function and 

```python
agent.test()
agent.visualization(5) # input the number of shape to visualize
```

comment `agent.train()`, then run

```python
python main.py
```




## See latent interpolation result

We define the `interpolation`  also in `main.py`, to use it, uncomment `agent.interpolation()`, to interpolate between specific shape, input the shape ID in dataset, like `agent.interpolation(177, 1309)`



## Pretrained weight

We provide pretrained weight for test and interpolation. Download the weight and create a new folder `weight` in the same path as `main.py`, and put the weight into `weight`.

Weight for baseline [link](https://drive.google.com/file/d/1fAbHVaasBZPvFCdS0gxepXuJBkfj_WH6/view?usp=sharing)

Weight for soft boundary [link](https://drive.google.com/file/d/1WHPNrthvsD5-3xWdogG-i1C77ho8GrSj/view?usp=sharing)




## Visualize loss curves

We support tensorboard to visualize loss curves when training, run

```
tensorboard --logdir event --port 6008
```

and open the link `localhost:16008` in explorer to see the training curves.



## Example Result

![](https://github.com/jianghd1996/IM-Net_reproduction/blob/master/result/result.JPG)

(a) is groundtruth, (b) is result of baseline, (c) is result of soft boundary, the noise is much less in soft boundary



![](https://github.com/jianghd1996/IM-Net_reproduction/blob/master/result/interpolation.JPG)

Interpolation result, (a) is result of baseline, (b) is result of soft boundary, with smooth operation, the noise is filtered in (a). The detail is better in (b) in the third and fifth chairs (from left to right)
