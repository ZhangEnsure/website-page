# MobileNet V1

适用于移动端本地进行实时边缘计算，不需要上传到云服务器中。一来是保障了实时性，二是保障了一些隐私的信息。在终端需要更大的计算能力，同时我们的模型也要更加的轻量化，这就对我们的软硬件都提出了较高的要求。端云结合，云端训练模型，得到的模型部署在终端。那么轻量化网络可以优化的具体方向有哪些呢？例如对深度学习框架进行CUDA加速、因为轻量化网络比较精简，所以该模型会不会被欺骗？这就引入了对抗学习。
<img src="https://s3.bmp.ovh/imgs/2022/11/02/30081a842eefb9be.jpg" class='img-fluid' style="width:700px; margin:auto; display:block"/>

## Depthwise separable convolution

MobileNet 基于**深度可分离卷积**对卷积层进行优化和改进，从而降低参数量减少计算量，使网络更加轻量化，这是 MobileNet 核心思想。深度可分离卷积的图示如下：

<img src="https://s3.bmp.ovh/imgs/2022/11/02/0b8a2988d3cb9a24.jpg" class='img-fluid' style="width:500px; margin:auto; display:block"/>

正常的卷积是卷积核是$w*h*c_{in}*c_{out}$，在这张图片的$c_{in}$是3。每一次卷积都生成 feature_map 中的一个值，最后将 $c_{out}$ 个 feature_map 进行叠加得到卷积结果。但是，深度可分离卷积是每一个卷积核负责一个 channel 的卷积操作，生成一个 feature_map ，最后将卷积的结果进行叠加。再举一个 depthwise conv 例子如下：

<img src="https://s3.bmp.ovh/imgs/2022/11/02/8087c9bf750ff82c.jpg" class='img-fluid' style="width:500px; margin:auto; display:block"/>

一个组合起来的例子，先进行 depthwise conv 后进行 pointwise conv。后面的卷积就是正常的卷积操作。depthwise conv处理的是单通道的长宽空间信息，不关心跨通道的信息。后面的 1@1 pointwise conv 不关心长宽空间信息，只关心跨通道的信息。这样一来，我们把空间信息和跨通道信息进行解耦。概括一下下面的图，深度可分离卷积的结果由深度卷积depthwise conv和逐点卷积pointwise conv共同计算得到。

<img src="https://s3.bmp.ovh/imgs/2022/11/02/717fee650062cad0.jpg" class='img-fluid' style="width:500px; margin:auto; display:block"/>

再举一个例子详细阐述一下深度可分离卷积的计算分解步骤：

<img src="https://s3.bmp.ovh/imgs/2022/11/02/ea17bfe8ebf89f9e.jpg" class='img-fluid' style="width:500px; margin:auto; display:block"/>

## 深度可分离卷积的计算量和参数量分析

标准卷积乘法计算量：$(D_{K} \cdot D_{K} \cdot M) \cdot( N \cdot D_{F} \cdot D_{F})$，后面的括号代表着一共进行了多少次卷次操作，feature_map 中一个数值就对应着一次卷积核的卷积过程，一个 feature_map 中共有 $D_{F} \cdot D_{F}$ 次卷积操作，共有 $N$ 个 feeature_map，所以一共进行了 $N \cdot D_{F} \cdot D_{F}$ 次卷积。
