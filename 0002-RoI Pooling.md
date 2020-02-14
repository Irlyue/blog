# RoI Pooling

这篇博客着重从源码层面讲解一下RoI Pooling是怎么实现的，包括前向和后向操作。因为笔者自己实现的时候还是使用的Tensorflow框架，所以下面都以TF为准，但很迁移到其他主流框架。

## 原理

**RoI Pooling操作支持将不同大小的输入池化为固定大小的特征输出。**下面的动图（图片取自[这里]( https://deepsense.ai/region-of-interest-pooling-explained/ )）很好地描述了具体的操作流程

<center><img src="assets/0002-0.gif"/></center>
基本流程就是

1. **分窗**， 上图里面分2*2=4个子窗口；
2. **池化**，如果我们用最大池化，那么就是取每个子窗口的最大值，最终得到$2\times 2$的输出特征

## 前向传播
前向传播的时候涉及的输入参数
- 特征张量：$B\times H\times W\times C$，由于使用的是TF，所以channel通道在最后面
- 候选框$R$：$N\times 5$，表明有N个候选框，第一个元素`R[:, 0]`表明当前候选框对应batch里的哪张图片，取值为[0, B)
- 输出维度$w_p$和$h_p$：Fast RCNN里用的是7*7的输出维度。

输出参数有两个：
- 池化的输出特征$O$：$N\times h_p\times w_p\times C$
- 下标矩阵：$N\times h_p\times w_p\times C$表明$O$中的元素取自输入张量的哪个元素。

接着我们先来看CPU的实现版本，CUDA的实现非常类似，只有一行循环控制不一样：
```c++
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace std;


#define idx4_1(idx, d1, d2, d3, d4) ((idx / d4 / d3 / d2) % d1)
#define idx4_2(idx, d1, d2, d3, d4) ((idx / d4 / d3) % d2)
#define idx4_3(idx, d1, d2, d3, d4) ((idx / d4) % d3)
#define idx4_4(idx, d1, d2, d3, d4) (idx % d4)
#define tuple_to_one(idx1, idx2, idx3, idx4, d1, d2, d3, d4) (idx1*d2*d3*d4+idx2*d3*d4+idx3*d4+idx4)


REGISTER_OP("RoiPooling")
    .Input("input: float32")
    .Input("rois: int32")
    .Attr("pool_height: int")
    .Attr("pool_width: int")
    .Output("output: float32")
    .Output("indices: int32");


int force_within(int x, int left, int right){
    return min(max(x, left), right);
};

void RoiPoolingKernelLauncher(const float* input,
                              const int* rois,
                              int n_rois,
                              int n_channels,
                              int height,
                              int width,
                              int pool_height,
                              int pool_width,
                              float* output,
                              int* indices);

class RoiPoolingOp: public OpKernel{
    private:
        int pool_height;
        int pool_width;
    public:
        explicit RoiPoolingOp(OpKernelConstruction* context): OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("pool_height", &pool_height));
            OP_REQUIRES_OK(context, context->GetAttr("pool_width", &pool_width));
        }

        void Compute(OpKernelContext* context)override{
            const Tensor& input_tensor = context->input(0);
            const Tensor& rois_tensor = context->input(1);

            auto input = input_tensor.flat<float>();
            auto rois = rois_tensor.flat<int32>();

            Tensor* output_tensor = NULL;
            Tensor* indices_tensor = NULL;

            auto input_shape = input_tensor.shape();   // [batch_size, height, width, n_channels]
            auto rois_shape = rois_tensor.shape();     // [batch_size, top, left, bottom, right]

            int n_rois       = rois_shape.dim_size(0);
            int input_height = input_shape.dim_size(1);
            int input_width  = input_shape.dim_size(2);
            int n_channels   = input_shape.dim_size(3);

            TensorShape output_shape = TensorShape({static_cast<int64>(n_rois),
                                                    static_cast<int64>(pool_height),
                                                    static_cast<int64>(pool_width),
                                                    static_cast<int64>(n_channels)});

            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &indices_tensor));

            auto output = output_tensor->flat<float>();
            auto indices = indices_tensor->flat<int32>();

            RoiPoolingKernelLauncher(input.data(),
                                     rois.data(),
                                     n_rois,
                                     n_channels,
                                     input_height,
                                     input_width,
                                     pool_height,
                                     pool_width,
                                     output.data(),
                                     indices.data());
        }
};


void RoiPoolingKernelLauncher(const float* input,
                              const int* rois,
                              int n_rois,
                              int n_channels,
                              int height,
                              int width,
                              int pool_height,
                              int pool_width,
                              float* output,
                              int* indices){
    int N = n_rois * pool_height * pool_width * n_channels;
    for(int i = 0; i < N; i++){
        // (n, h, w, c) indexed into the output tensor
        int n = idx4_1(i, n_rois, pool_height, pool_width, n_channels);
        int h = idx4_2(i, n_rois, pool_height, pool_width, n_channels);
        int w = idx4_3(i, n_rois, pool_height, pool_width, n_channels);
        int c = idx4_4(i, n_rois, pool_height, pool_width, n_channels);

        int roi_batch_idx = rois[n*5];
        int roi_top       = rois[n*5 + 1];
        int roi_left      = rois[n*5 + 2];
        int roi_bottom    = rois[n*5 + 3];
        int roi_right     = rois[n*5 + 4];
        int roi_height = max(1, roi_bottom - roi_top + 1);
        int roi_width  = max(1, roi_right - roi_left + 1);

        float bin_h = static_cast<float>(roi_height) / pool_height;
        float bin_w = static_cast<float>(roi_width) / pool_width;

        int hstart = static_cast<int>(floor(h * bin_h));
        int wstart = static_cast<int>(floor(w * bin_w));
        int hend   = static_cast<int>(ceil((h+1) * bin_h));
        int wend   = static_cast<int>(ceil((w+1) * bin_w));

        // force the index within range
        hstart = force_within(hstart + roi_top, 0, height);
        hend   = force_within(hend + roi_top, 0, height);
        wstart = force_within(wstart + roi_left, 0, width);
        wend   = force_within(wend + roi_left, 0, width);


        bool is_empty = (hend <= hstart) || (wend <= wstart);

        float maxval = is_empty ? 0: -99999999.0;
        int maxidx = -1;

        // loop over the (hstart, wstart, hend, wend) sub-window
        // record the maximum value and the related index
        // the max pooling can be easily replace with other strategy(say average pooling)
        for(int idx_h = hstart; idx_h < hend; idx_h++){
            for(int idx_w = wstart; idx_w < wend; idx_w++){
                int input_idx = tuple_to_one(roi_batch_idx, idx_h, idx_w, c, 0, height, width, n_channels);
                if(input[input_idx] > maxval){
                    maxval = input[input_idx];
                    maxidx = input_idx;
                }
            }
        }
        output[i] = maxval;
        indices[i] = maxidx;
    }
};


REGISTER_KERNEL_BUILDER(Name("RoiPooling").Device(DEVICE_CPU), RoiPoolingOp);
```

逐行来看代码，前面先是一些包含头文件的语句，接着定义了几个宏指令方便计算下标
```c++
#define idx4_1(idx, d1, d2, d3, d4) ((idx / d4 / d3 / d2) % d1)
#define idx4_2(idx, d1, d2, d3, d4) ((idx / d4 / d3) % d2)
#define idx4_3(idx, d1, d2, d3, d4) ((idx / d4) % d3)
#define idx4_4(idx, d1, d2, d3, d4) (idx % d4)
#define tuple_to_one(idx1, idx2, idx3, idx4, d1, d2, d3, d4) (idx1*d2*d3*d4+idx2*d3*d4+idx3*d4+idx4)
```
我们知道多维张量在内存里其实是以一维连续数组的形式存在的，实现的时候循环遍历每个输出位置，就需要很方便地进行1维坐标跟4维坐标之间的相互转换。即实现`(i1, i2, i3, i4)`的4维坐标转换第`i`个元素的1维坐标。

接下来很多模版代码，可以跳过直接看93行的`RoiPoolingKernelLauncher`函数。103行定义了总共会有多少个输出元素，也就是接下来的循环次数。每个`i`对应一个位置，我们需要把`i`转换为对应的四维坐标。111-117行代码提取候选框的信息。接着119-120这两行确定每个子窗口的大小，注意这里用的是浮点类型。接着计算这个窗口的起始和终止区域，并且保证不会超出图片区域。特别特殊的情况下，可能出现子窗口的大小为0的情况，这种情况就直接返回0作为池化的结果。

接着两层for循环遍历子窗口的每个元素，记录下最大值极其出现的位置，循环结束后保存这个结果。

上述过程循环直至每个输出位置都遍历完毕。

接下来说一下怎么利用CUDA以进行GPU加速，其实就一行代码不同（当然还包括其他一些必要的模块代码），只需要把104行的`for(int i = 0; i < N; i++){`替换为`CUDA_KERNEL_LOOP(i, N) {`即可。这里`CUDA_KERNEL_LOOP`是CUDA工具集提供的一个宏指令，这样for循环是可以加速并行的，同时计算多个输出位置。

## 后向传播
后向传播非常简单，从上游来会有一个导数`dout`，这个张量跟RoI Pooling的池化输出具有相同的维度。我们仍然只需`CUDA_KERNEL_LOOP(i, N)`循环`dout`的每个位置，利用前向传播记录的下标把这个导数加到对应的位置。

## 总结与思考
上述就是RoI Pooling操作的实现过程，真正进行有意义计算的代码应该就一百来行。从这些技术细节我们还可以得出一些有趣的结论：

**其实输入矩阵以$B\times C\times H\times W$的格式（即通道维度在前）会带来更快的计算速度**。因为在RoI Pooling中我们经常都是需要访问平面内一个窗口内的元素的，如果以$B\times C\times H\times W$的形式输入，会具有更好的局部性(Locality)，减少cache miss。相反，$B\times H\times W\times C$则导致元素按通道依次排列，但这里我们很少沿着通道维度去依次访问，所以cache命中率会大大降低。

同样，上述代码如果要迁移到Pytorch里面主要就是考虑张量元素是如何排布的，其他都基本一致。