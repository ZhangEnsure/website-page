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

> 标准卷积

乘法计算量：
$$(D_{K} \cdot D_{K} \cdot M) \cdot( N \cdot D_{F} \cdot D_{F})$$
$ N \cdot D_{F} \cdot D_{F}$ 代表着一共进行了多少次卷次操作。feature_map 中一个数值就对应着一次卷积核的卷积过程，一个 feature_map 中共有 $D_{F} \cdot D_{F}$ 次卷积操作，共有 $N$ 个 feeature_map，所以一共进行了 $N \cdot D_{F} \cdot D_{F}$ 次卷积。一次卷积需要 $D_{K} \cdot D_{K} \cdot M$ 个乘法操作。

参数量：$D_{K} \cdot D_{K} \cdot M \cdot N$

> 深度可分离卷积

乘法计算量：
$$(D_{K}\cdot D_{K}) \cdot (D_{F}\cdot D_{F}\cdot M)+(M*1*1)\cdot (D_{F}\cdot D_{F}\cdot N)$$
参数量：$M \cdot D_{K}\cdot D_{K}+M \cdot N$

在深度可分离网络中引入了两个超参数 $\alpha$ 宽度超参数，用来控制网络的宽度（通道数的 M 和 N 都乘以了这个超参数）；$\rho$ 分辨率超参数，控制输入图像的尺寸，进而控制中间层 feature_map 的大小，最后得到公式：
$$D_{K} \cdot D_{K} \cdot \alpha M \cdot \rho D_{F} \cdot \rho D_{F}+\alpha M \cdot \alpha N \cdot \rho D_{F} \cdot \rho D_{F}$$

> 标准卷积与深度可分离卷积的对比

乘法计算量：

$$ \frac{D_{K} \cdot D_{K} \cdot M \cdot D_{F} \cdot D_{F}+M \cdot N \cdot D_{F} \cdot D_{F}}{D_{K} \cdot D_{K} \cdot M \cdot N \cdot D_{F} \cdot D_{F}} = \frac{1}{N}+\frac{1}{D_{K}^{2}}$$

参数量：

$$\frac{D_{k} \cdot D_{k} \cdot M+M \cdot N}{D_{k} \cdot D_{k} \cdot M \cdot N}=\frac{1}{N}+\frac{1}{D_{k}^{2}}$$

## MobileNet V1网络结构

<img src="https://s3.bmp.ovh/imgs/2022/11/02/13d04f70e538bdcc.jpg" class='img-fluid' style="width:500px; margin:auto; display:block"/>

全局网络图，这里有 padding，所以有的计算可能不是很直观。

<img src="https://s3.bmp.ovh/imgs/2022/11/02/8cb50d348ebb2249.jpg" class='img-fluid' style="width:500px; margin:auto; display:block"/>

## 计算性能分析

通过深度可分离卷积确实减少了参数量和计算量，下面分析两个计算参数的占比，可以发现1@1卷积资源消耗占比很大，我们尝试着优化。
<img src="https://s3.bmp.ovh/imgs/2022/11/02/e654ff91d1eea4c0.jpg" class='img-fluid' style="width:500px; margin:auto; display:block"/>

优化的策略是把卷积核拉成一个行向量，不同的卷积核的行向量进行叠加。把图像的每一次卷积的感受野拉成一个列向量，每个感受野的列向量进行列叠加。最后进行矩阵的点乘。在 MobileNet V1 中的 1@1 卷积天然就是一个排列好的向量，所以运算速度较快。
<img src="https://s3.bmp.ovh/imgs/2022/11/02/559809ecd09a34ea.jpg" class='img-fluid' style="width:500px; margin:auto; display:block"/>
