# **第三届中国AI+创新创业大赛：半监督学习目标定位竞赛第六名方案**

团队名称：逍遥郎1392

团队成员：谢自强

团队成绩：决赛第六名

模型结果：https://pan.baidu.com/s/1m4kHFZnF-5WU51DFhzvMQA， a3h7

aistudio地址：https://aistudio.baidu.com/aistudio/projectdetail/2173012

## **1. 简介**

任务：比赛要求基于少量有标注数据训练模型，使分类网络具有目标定位能力，实现半监督目标定位任务。每一位参赛选手仅可以使用ImageNet大型视觉识别竞赛(ILSVRC)的训练集图像作为训练数据，其中有标注的训练数据仅可以使用大赛组委会提供的像素级标注数据。

环境工程：Python3.7+PaddlePaddle2.1.0，PaddleSeg

数据集：
训练数据集包括50,000幅像素级有标注的图像，共包含500个类，每个类100幅图像；
A榜测试数据集包括11,878幅无标注的图像；
B榜测试数据集包括10,989幅无标注的图像。
本方案仅使用了提供的标注数据，未使用没有标注的数据。

## **2. 方案**

数据标注均是像素级的标注，因此可以采用语义分割的方法，实现目标的定位。总体方案思路为单模型优化+单模型融合+多模型融合。

### **2.1 单模型**

采用语义分割效果比较好的deeplabv3和deeplabv3p两种方法，分别进行优化训练。

1） backbone选择resnet50_vd和resnet101_vd，但是resnet101的效果反而不如resnet50，最终放弃；

2） 分辨率采用256x256，416x416，512x512，其中以256为主，更大的分辨率仅用于微调，并且只训练了一个模型，采用更大的分辨率有助于对目标边界和一些细节进行准确的判断；

3） 训练数据，采用了交叉验证的思路，分别抽出了2个10000张图片作为评估集，然后分别将剩下的40000张作为训练集，另外又采用了全部数据进行训练，相当于有了三个不完全相同的训练集，不同的训练数据可以让模型学习到不同的特征，丰富模型的多样性，有助于模型融合时提升效果；

4） 数据增强，采用了随机上下、左右翻转，数据抖动和随机选转；

### **2.2 模型融合**

1）单模型融合

在预测时添加aug_pred和flip_vertical即可实现图像正常预测和垂直翻转预测的融合结果；

2）多模型融合

取多个效果较好的模型的预测结果，进行投票，采用少数服从多数的原则，得到最终的结果；

## **3. 结果复现**

所有操作均在work/PaddleSeg文件夹下；

所有配置文件均在configs/ssc中，所有模型均在output中，与配置文件同名，另外考虑到工程大小，所有模型均只保留了一个；

分别执行预测命令，并添加aug_pred和flip_vertical参数，得到各个模型的预测结果，然后进行投票即可，投票脚本为vote.py；

注：将模型文件解压后放入PaddleSeg文件夹下，与配置文件同名，最终融合的结果在PaddleSeg文件夹下，压缩包文件名为pred_0.77813.zip；


```python
! cd work/PaddleSeg/; python predict.py --config configs/ssc/deeplabv3p_resnet50_vd_ft.yml --model_path output/deeplabv3p_resnet50_vd_ft/iter_60000/model.pdparams --image_path ../data/test_image --save_dir result_b/dlv3p_r50_ft_vertical --aug_pred --flip_vertical
```


```python
! cd work/PaddleSeg/; python predict.py --config configs/ssc/deeplabv3_resnet50_vd.yml --model_path output/deeplabv3_resnet50_vd/iter_120000/model.pdparams --image_path ../data/test_image --save_dir result_b/dlv3_r50_vertical --aug_pred --flip_vertical
```


```python
! cd work/PaddleSeg/; python predict.py --config configs/ssc/deeplabv3p_resnet50_vd_fold.yml --model_path output/deeplabv3p_resnet50_vd_fold/iter_120000/model.pdparams --image_path ../data/test_image --save_dir result_b/dlv3p_r50_fold_vertical --aug_pred --flip_vertical
```


```python
! cd work/PaddleSeg/; python predict.py --config configs/ssc/deeplabv3_resnet50_vd_416.yml --model_path output/deeplabv3p_resnet50_vd_416/iter_120000/model.pdparams --image_path ../data/test_image --save_dir result_b/dlv3p_r50_416_vertical --aug_pred --flip_vertical
```


```python
! cd work/PaddleSeg/; python predict.py --config configs/ssc/deeplabv3_resnet50_vd_aug.yml --model_path output/deeplabv3p_resnet50_vd_aug/iter_160000/model.pdparams --image_path ../data/test_image --save_dir result_b/dlv3p_r50_aug_vertical --aug_pred --flip_vertical
```


```python
! cd work/PaddleSeg/; python predict.py --config configs/ssc/deeplabv3p_resnet50_vd_512_all.yml --model_path output/deeplabv3p_resnet50_vd_512_all/iter_120000/model.pdparams --image_path ../data/test_image --save_dir result_b/dlv3p_r50_512_all_vertical --aug_pred --flip_vertical
```


```python
! cd work/PaddleSeg/; python predict.py --config configs/ssc/deeplabv3p_resnet50_vd_all.yml --model_path output/deeplabv3p_resnet50_vd_all/iter_200000/model.pdparams --image_path ../data/test_image --save_dir result_b/dlv3p_r50_all_vertical --aug_pred --flip_vertical
```


```python
! cd work/PaddleSeg/; python predict.py --config configs/ssc/deeplabv3_resnet50_vd_all.yml --model_path output/deeplabv3_resnet50_vd_all/iter_200000/model.pdparams --image_path ../data/test_image --save_dir result_b/dlv3_r50_all_vertical --aug_pred --flip_vertical
```


```python
! cd work/PaddleSeg/; python predict.py --config configs/ssc/deeplabv3p_resnet50_vd_aug_all_fold.yml --model_path output/deeplabv3p_resnet50_vd_aug_all_fold/iter_200000/model.pdparams --image_path ../data/test_image --save_dir result_b/dlv3p_r50_aug_all_fold_vertical --aug_pred --flip_vertical
```


```python
! cd work/PaddleSeg; python vote.py
```


```python
! cd work/PaddleSeg;  zip -roq results pred.zip
```

## **4. 模型训练**

所有操作均在work/PaddleSeg文件夹下,所有配置文件均在configs/ssc中，所有参数配置等均保持与比赛期间一致，执行训练命令，分别训练各个模型即可。
以下列出一个模型的训练，其他模型只需替换配置文件和保存路径即可。


```python
! cd work/PaddleSeg/; python train.py --config configs/ssc/deeplabv3p_resnet50_vd.yml --do_eval --use_vdl --save_interval 10000 --log_iters 20 --save_dir output/deeplabv3p_resnet50_vd
```

## **5. 其他方案**

尝试了其他多种方法，但是没有取得效果，另外由于参与时间较晚，有一些想法没有进行尝试；

### **5.1 数据增强**

对增加了90°的旋转，得分降低；

为了避免resize导致目标的比例发生变化，将所有图片补全为1：1，分别采用了位置随机填充和中心固定填充，得分降低；

增加了随机crop，得分降低；

### **5.2 模型**

采用了unet、unet3p、pspnet、ocrnet等方法效果都不好；

### **5.3 backbone**

基本网络选择了resnet50，而采用resnet101的效果反而不如resnet50，另外测试过resnet152等更大的基础网络，收敛比较慢，最终放弃；

### **5.4 模型融合**

采用模型预测的softmax得分求和的软投票方法，但是效果反而不如直接进行预测结果的vote方法；

### **5.5 后处理**

采用crf等后处理方法，没有提升；

### **5.6 未尝试的方案**

1）拆分合并类别
数据实际包含了500个类别，各个类别相差还是比较大，因此可以考虑按照一定的规则合并一些类别，比如动物可以归为一类，或者也可以拆分为野生动物和家禽等，拆分后虽然单类的数据相比只分一类变少，但是类间差异减小，更有利于网络进行学习；

2） 基于合并的类别，可以考虑多分类或者为每个类别单独训练一个二分类；

3） 不同的优化器、学习率策略和loss函数，由于时间关系未做横向对比；

4） Transformer等新方法；
