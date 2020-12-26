# transformer_noise
使用不同noise得到的单位时间序列的准确率
=
应用于单维时间序列 score矩阵为channel*channel 扩充且附加noise
<table>
  <tr>
    <th>hyperparameters</th> <th>d_channel</th> <th>d_model</th> <th>Query size</th> <th>Value size</th> <th>Number of heads</th> <th>Number of encoder</th> <th>dropout</th> <th>Positional encoding</th> <th>mask</th>
  </tr>
  <tr>
    <th>value</th> <th>32</th> <th>512</th> <th>8</th> <th>8</th> <th>6</th> <th>6</th> <th>0.2</th> <th>True</th> <th>False</th>
  </tr>
</table>

=
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
