# reformer-chinese

使用reformer训练续写

GTP2的参数越来越大，显存实在是开销不起，还好Google又开源了reformer，让我们用小的显存也可以做生成了。

reformer可以做的东西很多，

## 训练

```
python3 train.py --epochs 1 --device cpu --batch_size 320 --gradient_accumulation 1 --lr 0.01 --num_pieces 10 --depth 6  --full_attn_thres 128 --dim 128  --stride 60 --pretrained_model  model/
```
### 参数
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

模型输出到model目录下

## 数据

数据放在/data目录下命名为train.txt的纯文本即可
一行一段


## 测试 




#### 示例1 知识提取

尝试使用reformer做知识提取，先看一看效果
使用百度AI开源的知识提取数据做的训练。下面是效果

> 原始语句： 《你的嘴》收录于歌手金莎的音乐专辑《星月神话》，由许嵩作词作曲，2010年10月15日首发 

> 正确知识：   [KG] 你的嘴,所属专辑,星月神话 [/KG]  [KG] 你的嘴,作词,许嵩 [/KG]  [KG] 你的嘴,作曲,许嵩 [/KG]  [KG] 你的嘴,歌手,金莎 [/KG] 

> 提取知识：[['你的嘴', '所属专辑', '星月神话'], ['你的嘴', '作曲', '许嵩'], ['你的嘴', '作曲', '许嵩'], ['你的嘴', '歌手', '金莎']]


当然这是其中比较好的，大体还是不错的。

和使用mark模式不同的是，使用生成模型做训练的结果是可以提取出原文不存在的一些知识。比如所属专辑这种，不是原文中存在的文字，显然使用mark模式是做不到的。
不过生成模型也有时候会出现太过放飞自我的问题，比如出现不存在的名字或者实体等等。


## 感谢

1. reformer-pytorch - https://github.com/lucidrains/reformer-pytorch


## ME
我的博客：
https://www.terrychan.org/
