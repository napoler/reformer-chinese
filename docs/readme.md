# reformer-chinese

使用reformer训练续写

GTP2的参数越来越大，显存实在是开销不起，还好Google又开源了reformer，让我们用小的显存也可以做生成了。

```
python3 train.py --epochs 1 --device cpu --batch_size 320 --gradient_accumulation 1 --lr 0.01 --num_pieces 10 --depth 6  --full_attn_thres 128 --dim 128  --stride 60 --pretrained_model  model/
```

- dim ：窗口大小
- stride ：滑窗大小
- pretrained_model ：上次训练的模型
- device：cpu/cuda
- epochs：迭代次数
- batch_size：单次样本数目
- gradient_accumulation：梯度累积
- --raw ：是否进行预处理 
- num_pieces：分片数目
- depth：
- full_attn_thres：
- output_dir：输出的目录