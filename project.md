项目背景->模型架构->数据处理->训练过程->优化调整->结果->总结

# 无人驾驶目标检测

### 项目背景

在随着汽车智能化、电子化的推进，无人驾驶已经是未来汽车发展的必然趋势，利用计算机视觉相关技术可以检测出交通环境中的各个要素信息，比如行人、常见车辆、交通指示牌以及红绿灯，可以为车辆的规划与控制提供一定的帮助

### 模型架构

考虑到汽车在行驶中需要快速的识别道路上的物体，于是我选择了单阶段的端到端目标检测算法yolov5，yolov5在保证速度的前提下又有着十分不错的精度；

[![Rg4KII.png](https://z3.ax1x.com/2021/07/03/Rg4KII.png)](https://imgtu.com/i/Rg4KII)

采取**Mosaic数据增强**方法，通过随即缩放、随机剪裁和随机排布的方式进行拼接，增强了小目标的检测效果；另外采用自适应锚框计算，自适应的计算不同训练集中的最佳锚框值；

Backbone，引入Focus结构，关键切片操作，切完再concat；采用CSPDarknet53，由卷积层和X个Res unit模块concat组成，采用CSP模块先将基础层的特征映射划分为两部分，然后通过跨阶段层次结构将它们合并，在减少了计算量的同时可以保证准确率；整体来看，主要有三方面的优点：1.增强了CNN的学习能力，使得轻量化的同时保持准确性；2.降低了计算瓶颈；3.降低了内存成本；采用**Leaky relu**激活函数：y = max(0, x) + leak*min(0,x) ，leak是一个很小的常数，这样保留了一些负轴的值，使得负轴的信息不会全部丢失；使用SPP模块，SPP的输入是512x20x20，经过1x1的卷积层后输出256x20x20，然后经过并列的三个Maxpool进行下采样，将结果与其初始特征相加，输出1024x20x20，最后用512的卷积核将其恢复到512x20x20。

Neck和yolov4中一样，都采用FPN+PAN的结构，FPN层的后面还添加了一个**自底向上的特征金字塔**，FPN层自顶向下传达**强语义特征**，而特征金字塔则自底向上传达**强定位特征**，两两联手，从不同的主干层对不同的检测层进行参数聚合，进一步提高特征提取的能力；同样加入CSP-net结构，加强了网络的特征融合能力；

输出端采用GIOU_Loss做Bounding box的损失函数，GIOU_Loss=1 - GIOU = 1- (IoU-差集/C)，ps：（IOU = 交集/并集）[![RZpLtS.png](https://z3.ax1x.com/2021/06/22/RZpLtS.png)](https://imgtu.com/i/RZpLtS)

**GIOU_Loss**在IOU的基础上，解决了边界框不重合时的问题，但仍无法区分相对位置关系，2020年AAAI提出了**DIOU_Loss**，考虑了**重叠面积**和**中心点距离**，当目标框包裹预测框的时候，直接度量2个框的距离，因此DIOU_Loss收敛的更快[![RZC5Zt.png](https://z3.ax1x.com/2021/06/22/RZC5Zt.png)](https://imgtu.com/i/RZC5Zt)

但没有考虑到长宽比，针对这个问题，又提出了CIOU_Loss（yolov4采用），CIOU_Loss和DIOU_Loss前面的公式都是一样的，不过在此基础上还增加了一个影响因子，将预测框和目标框的长宽比都考虑了进去[![RZCOMj.png](https://z3.ax1x.com/2021/06/22/RZCOMj.png)](https://imgtu.com/i/RZCOMj)



再来综合的看下各个Loss函数的不同点：

**IOU_Loss：**主要考虑检测框和目标框重叠面积。

**GIOU_Loss：**在IOU的基础上，解决边界框不重合时的问题。

**DIOU_Loss：**在IOU和GIOU的基础上，考虑边界框中心点距离的信息。

**CIOU_Loss：**在DIOU的基础上，考虑边界框宽高比的尺度信息。

Yolov5中采用加权[nms](https://zhuanlan.zhihu.com/p/50126479)方式

loss = GIOU_loss + cls_loss + obj_loss，使用**二进制交叉熵**和 **Logits** 损失函数计算类概率和目标得分的损失

### 数据

考虑到主要场景是针对行驶的道路，所以我采用bdd100k自动驾驶数据集，这是最大的开放式驾驶视频数据集之一，该数据集具有地理，环境和天气多样性，从而能让模型能够识别多种场景，具备更多的泛化能力；使用2D矩形框共标注了10万张图像，标注对象包括公共汽车、交通灯、交通标志、人、自行车、卡车、汽车、电动车、火车和骑手；

Bdd100k的标签是由Scalabel生成的JSON格式，首先得将bdd100k的标签转换为coco格式，然后再将coco格式转换为yolo格式

### 训练

训练了200个epoch，大约在180个epoch左右就达到了最优效果，[mAP_0.5](https://blog.csdn.net/luke_sanjayzzzhong/article/details/89851944) = 41.3%

### 优化

可以调整骨干网络，换用更加轻量级的mobilenet；通过h['fl_gamma']参数开启**[focal Loss](https://www.cnblogs.com/king-lps/p/9497836.html)**,默认配置没有采用focal loss

### 总结

# 瓷砖表面瑕疵质检

### 项目背景

瓷砖表面的瑕疵检测是瓷砖行业生产和质量管理的重要环节，也是困扰行业多年的技术瓶颈。利用高效可靠的计算机视觉算法，尽可能快与准确的给出瓷砖疵点具体的位置和类别;

### 模型架构

见首个项目

[yolov5深度解析](https://zhuanlan.zhihu.com/p/183838757)

### 数据

切图：原图大小为:h=6000,w=8192；切图的大小为:640x640,剔除纯背景图片， overlap比例:0.2；步长为512；从原图左上角开始切图,切出来图像的左上角记为x,y；y依次为:0,512,1024,....,5120.但接下来却并非是5632,因为5632+640>6000,所以这里要对切图的overlap做一个调整,最后一步的y=6000-640；

进行简单的数据增强：灰度正规化、增强对比度、水平垂直翻转，效果略有提升

### 训练

batch_size为16，优化器为SGD，LambdaLR自定义调整学习率，训练了200个epoch，发现在150个epoch左右达到最优效果，0.2[ACC](https://www.pianshen.com/article/6386546945/)+0.8mAP = 0.5357

### 难点

数据存在目标类别数量不均衡，尺度不均衡，小目标瑕疵占比较大，瑕疵的大小变化大；

后面优化是通过h['fl_gamma']参数开启**focal Loss**，效果略有提升；

### 展望

可以尝试该换backbone例如Res2net；加入[DCN](https://blog.csdn.net/yeler082/article/details/78370795)模块，双阈值策略等等

# 遥感影像地物要素分割

### 项目背景

天池-2021全国数字生态创新大赛-智能算法赛-生态资产智能分析

基于不同地形地貌的高分辨率遥感影像资料，识别提取土地覆盖和利用类型，实现生态资产盘点、土地利用动态监测、水环境监测与评估、耕地数量与监测等应用；

### 模型架构

考虑到分割的精度要求，采用[Unet++](https://www.zhihu.com/column/p/295427213?utm_medium=social&utm_source=weibo)网络架构，使用[efficient-b6](https://zhuanlan.zhihu.com/p/137191387)作为backbone；

loss是[focal loss](https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&tn=baidu&wd=focal-loss&oq=dice-loss&rsv_pq=9491f9510000baa9&rsv_t=2cdaGaYVlktSJmlyD1koJiymN%2F%2FbkuUSOx5%2Fho90VjHlwk2muzpc%2FRHvbpo&rqlang=cn&rsv_dl=tb&rsv_enter=1&rsv_sug3=6&rsv_sug1=3&rsv_sug7=100&rsv_sug2=0&rsv_btype=t&inputT=927&rsv_sug4=1932)+[dice-loss](https://zhuanlan.zhihu.com/p/269592183)+[softCrossEntropy](https://www.jianshu.com/p/47172eb86b39)联合loss，优化器用的是[AdamW](https://zhuanlan.zhihu.com/p/113112032)，[warm up](https://zhuanlan.zhihu.com/p/261312302) [Cosine](https://zhuanlan.zhihu.com/p/93624972) Scheduler动态调整学习率；[学习率调整策略](https://zhuanlan.zhihu.com/p/93624972)

### 数据

- 先将图片转换为.jpg格式
- 数据增强：对训练集一定概率（0.5）地进行以下操作的组合与随机选择：水平、竖直翻转，旋转90°，转置;
- 对训练集进行归一化操作(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),验证集和测试集也同样进行这个归一化操作；
- 划分数据集，验证集训练集的20%，训练采用的是全数据；

### 训练

batch_size为64，lr = 3e-4，训练了300个epoch，模型大约在250个epoch左右收敛到最优；使用通用指标平均交并比mIoU，计算每个类别的交并比的平均值，仅对算法效果进行评价，具体计算公式为：

[![R8Shvt.png](https://z3.ax1x.com/2021/06/26/R8Shvt.png)](https://imgtu.com/i/R8Shvt)

[mIoU](https://zhuanlan.zhihu.com/p/88805121)最好得分为0.3829

### 优化

开始用loss的是softCrossEntropy，因为数据存在类别不平衡问题，改进loss加入focal-loss和dice-loss，mIoU提升了约1个点；过程中还尝试过换模型deeplabv3+，更大的backbone，换SGD优化器，学习率调整策略等等，效果都不如这个好；还有一些trick没有尝试，比如多模型融合，多尺度训练/测试等等



# 经典网络解析

[AlexNet、VGG、NIN、GoogLeNet、ResNet etc.](https://zhuanlan.zhihu.com/p/47391705)

[FCN、SegNet、Unet、PSPNet、DeepLab系列](https://blog.csdn.net/qq_37002417/article/details/108274404)

