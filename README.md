# bayes-classifier-on-image-segmentation

###　说明
1. 309.bmp是需要分割的图像，只要求对鱼的部分进行分割
2. array_sample.mat是用于训练的matlab格式的样本数据，其中每一行代表一个样本信息，第1列为其灰度值，第2-4列分别对应r,g,b颜色值，最后一列代表label信息，其中1代表属于第一类，-1代表属于第二类
3. Mask.mat为一个二值图像，通过源图像与该图像的点乘运算即可得到需要分割的目标，即nemo鱼的部分。

### 运行
* `python color_para_based.py` 实现彩色图像分割（效果较差）

* `python color_para_based.py` 实现灰度图像分割（效果较好）
