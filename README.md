# transformer_noise
使用不同noise得到的单位时间序列的准确率
=
应用于单维时间序列 score矩阵为channel*channel 扩充且附加noise

hyperparameters | d_channel | d_model | Query size | Value size | Number of heads | Number of encoder | dropout | Positional encoding | mask |
----------------|-----------|---------|-----------|-----------|-------------------|-------------------|---------|---------------------|------|
value | 32 | 512 | 8 | 8| 6 | 6 | 0.2 | True | Fasle |

---
数据集 | 测试集准确率(noise=0) | 测试集准确率(noise=0.2) | 测试集准确率(noise=0.4) | 测试集准确率(noise=0.6) |
-------|----------------------|------------------------|--------------------------|-------------------------|
ACSF1|66.0%|***70.0%***|69.0%||
Adiac|***77.44%***|62.56%|51.03%||
ArrowHead|***78.29%***|***78.29%***|***78.29%***||
Beef|***86.67%***|***86.67%***|83.33%||
BeetleFly|90.0%|90.0%|***95.0%***|***95.0%***|
BirdChicken|***85.0%***|***85.0%***|***85.0%***|***85.0%***|
BME|98.67%|***99.33%***|98.0%|98.0%|
Car|86.67%|86.67%|***88.33%***|81.67%|
CBF|94.0%|91.56%|94.0%|***96.67%***|
Coffee|***100%***|***100%***|***100%***|***100%***|

<table>
  <tr>
      <th>数据集</th>
      <th>测试集准确率(noise=0)</th>
      <th>测试集准确率(noise=0.2)</th>
      <th>测试集准确率(noise=0.4)</th>
      <th>测试集准确率(noise=0.6)</th>
  </tr>
  <tr>
      <th>ACSF1</th>  <th>66.0%</th> <th>70.0%</th> <th>69.0%</th> <th></th>
  </tr>
  <tr>
      <th>Adiac</th>  <th>77.44%</th> <th>62.56%</th> <th>51.03%</th> <th></th>
  </tr>
  <tr>
      <th>ArrowHead</th>  <th>78.29%</th> <th>78.29%</th> <th>78.29%</th> <th></th>
  </tr>
  <tr>
      <th>Beef</th>  <th>86.67%</th> <th>86.67%</th> <th>83.33%</th> <th></th>
  </tr>
  <tr>
      <th>BeetleFly</th>  <th>90.0%</th> <th>90.0%</th> <th>95.0%</th> <th>95.0%</th>
  </tr>
  <tr>
      <th>BirdChicken</th>  <th>85.0%</th> <th>85.0%</th> <th>85.0%</th> <th>85.0%</th>
  </tr>
  <tr>
      <th>BME</th>  <th>98.67%</th> <th>99.33%</th> <th>98.0%</th> <th>98.0%</th>
  </tr>
  <tr>
      <th>Car</th>  <th>86.67%</th> <th>86.67%</th> <th>88.33%</th> <th>81.67%</th>
  </tr>
  <tr>
      <th>CBF</th>  <th>94.0%</th> <th>91.56%</th> <th>94.0%</th> <th>96.67%</th>
  </tr>
  <tr>
      <th>Coffee</th>  <th>100%</th> <th>100%</th> <th>100%</th> <th></th>
  </tr>
  
</table>
