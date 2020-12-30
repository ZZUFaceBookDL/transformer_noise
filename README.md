# transformer_noise
使用不同noise得到的单位时间序列的准确率
=
应用于单维时间序列 score矩阵为channel*channel 扩充且附加noise

hyperparameters | d_channel | d_model | Query size | Value size | Number of heads | Number of encoder | dropout | Positional encoding | mask |
----------------|-----------|---------|-----------|-----------|-------------------|-------------------|---------|---------------------|------|
value | 32 | 512 | 8 | 8| 6 | 6 | 0.2 | True | Fasle |

---
数据集 | FCN | ResNet | 测试集准确率(noise=0) | 测试集准确率(noise=0.2) | 测试集准确率(noise=0.4) | 测试集准确率(noise=0.6) |
-------|-----|--------|----------------------|------------------------|--------------------------|-------------------------|
ACSF1|-|-|66.0%|***70.0%***|69.0%||
Adiac|***84.4%***|82.9%|77.44%|62.56%|51.03%||
ArrowHead|84.3%|***84.5%***|78.29%|78.29%|78.29%||
Beef|69.7%|75.3%|***86.67%***|***86.67%***|83.33%||
BeetleFly|86.0%|85.0%|90.0%|90.0%|***95.0%***|***95.0%***|
BirdChicken|***95.5%***|88.5%|85.0%|85.0%|85.0%|85.0%|
BME|-|-|98.67%|***99.33%***|98.0%|98.0%|
Car|90.5%|***92.5%***|86.67%|86.67%|88.33%|81.67%|
CBF|99.4%|***99.5%***|94.0%|91.56%|94.0%|96.67%|
Coffee|100.0%|***100.0%***|100%|100%|100%|100%|

针对单位数据集的模型实验结果总结
=
对三个单维时间序列进行实验 score=channel 

数据集 | FCN| ResNet | 本模型最好准确率(test/train) | 模型描述 | 本模型最好准确率(test/train) | 模型描述 |
-------|----|--------|-----------------------------|----------|------------------------------|---------|
ACSF1|-|-|70.0% / 93.0%|score=channel epoch=800 dropout=0.2 noise=0.2 channel=40 batch=8 PE=True Mask=False q=v=8 h=N=6|
Adiac|84.4%|82.9%|80.0% / 98.72%|score=channel epoch=1500 dropout=0 noise=0 channel=512 batch=8 PE=True Mask=True q=v=8 h=N=6|80.0% / 93.59%|score=channel epoch=1500 dropout=0.2 noise=0 channel=256 batch=32 PE=True Mask=True q=v=8 h=N=6|
ArrowHead|84.3%|84.5%|81.14% / 100%|gate+softmax epoch=500 dropout=0.7 noise=0 channel=32 batch=4 PE=True Mask=True q=v=8 h=N=4|
