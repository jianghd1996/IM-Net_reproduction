# IM-Net_reproduction
This is the project for course Deep Generation Model. In this project, I try to reproduce IM-Net in pytorch and optimize the training process.



##### Get started

---

- **Dataset**

We use dataset in [PartNet](https://cs.stanford.edu/~kaichun/partnet/) and voxelize the shape in $64^3$ resolution. To download our training data, please visit this [link](https://disk.pku.edu.cn:443/link/BB3144D411E61092DECED4C7F0C8ED11) (the voxelization is done by [Rundi Wu](https://github.com/ChrisWu1997/PQ-NET)).



- **Training**

In the `baseline` / `optimized_sampling` folder, run ```python data_preprocessing.py``` 


to do the post processing sampling on dataset and it will generate a folder `processed_data` under the `baseline` folder

Then run

```python
python main.py
CUDA_VISIBLE_DEVICES=0 python main.py # GPU user
```



- **Testing and visualization**

We define the `test()` and `visualization(number)` function in the object, to use them, run

```python
agent.test()
agent.visualization(5) # input the number of shape to visualize
```

in the main function of `main.py`



- **Visualize loss curves**

We support tensorboard to visualize loss curves when training, run

```
tensorboard --logdir event --port 6008
```

and open the link `localhost:16008` in explorer to see the training curves



Issue :

- [ ] Current sampling is time consuming
- [ ] Current results has many noise on the surface



To do list :

- [ ] Soft boundary
- [ ] Sampling method optimization