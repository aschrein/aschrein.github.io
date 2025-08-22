---
layout: post
title:  "Thinking about convolutions for graphics"

date:   2023-01-08 00:00:00 +0000
categories: jekyll update
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
      inlineMath: [['$$','$$']]
    }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Table of content
* this unordered seed list will be replaced
{:toc}

# Introduction

When introducing convolutions it does often start with something like this:

![](/assets/conv_for_gfx/wiki_conv.png)

[Source](https://en.wikipedia.org/wiki/Convolution)

or

![](/assets/conv_for_gfx/01_simple_convolution.jpg)

[Source](https://en.wikipedia.org/wiki/Convolution)

Which is fair but from a practical point of view it is not very helpful.

We need to add another dimension which is the feature dimension.
In graphics we usually work with 3D tensors, where the three dimensions correspond to width, height, and feature channels (e.g., RGB).
The default layout is referred to as HWC linear layout, which means that we're linearizing from the last dimension to the first e.g. it will be stored like [R G B R G B ..] in memory for RGB images. When working with ML frameworks the layout is usually BCHW which is batch_size, channels, height, width and in memory it might be stored as an array of 2D 'grayscale' single feature slices.

# Inference oriented point of view

For inference though as we care about the memory hieararchy and hw specific matrix multiplication instructions we want the features to be close to one another and multiple of the of the number of channels supported by the hardware(4, 8, 16 ..) which also makes thinking about the operations a bit easier, from my point of view. As all we do is just load some vector of values per pixel, multiply it by a matrix and then store that back - pretty straightforward compared to traditional shader workloads. So my point is that this pixel-feature-vector centered point of view makes it easier to think about common operations as we don't really care that much about the width and heights or batch size during inference. Also this helps when working with hw matrix multiplication instructions.

![](/assets/conv_for_gfx/conv_diagram.png)

In a naive implementation that would look something like that:

```c++
// Compute shader pseudocode
// T - quantized datatype that we use for storage and operations
void Conv1x1(i32x2 tid) {
  
  vector<T, N> input_features = load<T, N(input_texture, tid);
  matrix<T, M, N> weights     = load_weights<M, N>(conv1x1_weights);
  vector<f32, M> biases       = load<f32, M>(conv1x1_biases);

  // Matrix multiply
  vector<f32, M> output_features = matmul<T, M, N>(input_features, weights) + biases;

  // Convert back to the quantized domain
  vector<T, M> quantized = quantize<T, f32>(output_features);

  store<T, M>(output_texture, tid, quantized);
}

```

And 3x3 convolutions would look similar, with just extra concatenation:

```c++

// Compute shader pseudocode
// T - quantized datatype that we use for storage and operations
void Conv3x3(i32x2 tid) {
  
  vector<T, N * 9> input_features;

  // Load the 3x3 neighborhood of features
  // Or use any other im2col approach
  for (y in -1 .. 1) {
    for (x in -1 .. 1) {
      input_features[(y + 1) * 3 + (x + 1)] = load<T, N(input_texture, tid + /* offset */ i32x2(x, y));
    }
  }

  matrix<T, M, 9 * N> weights  = load_weights<M, 9 * N>(conv3x3_weights);
  vector<f32, M> biases        = load<f32, M>(conv3x3_biases);

  // Matrix multiply
  vector<f32, M> output_features = matmul<T, M, N>(input_features, weights) + biases;

  // Convert back to the quantized domain
  vector<T, M> quantized = quantize<T, f32>(output_features);

  store<T, M>(output_texture, tid, quantized);
}

```

And that's pretty much it. You have your operator implemented. Ofc there's other stuff like padding and stride, dilation but that can be an extension.

```python
# PyTorch convolution
conv3x3 = nn.Conv2d(in_channels=N, out_channels=M, kernel_size=3, stride=1, padding=1)

```

# Low rank operations

It's often useful as well to have lower rank operations, now that we're working in the matrix multiplication space.
For example, if we have a NxN matrix, the number of flops is O(N^3). But if we split that matrix into 2 smaller matrices that map half of the input features to another half of the output features, we'll get (N / 2)^3 * 2 which is 1/4 the original cost. Not hard to notice this has some nice properties, but the downside of of this is that during training the disjoint feature groups don't talk to one another, luckly we can solve that by adding another MxN matrix multiply after that to combine the features.

![](/assets/conv_for_gfx/conv_group_diagram.png)


Bonus my original sketch:

![](/assets/conv_for_gfx/sketch.png)


# Links

[1][wmma on RDNA3][1]

[1]: https://gpuopen.com/learn/wmma_on_rdna3/

[2][Using the Matrix Cores of AMD RDNA 4 architecture GPUs][2]

[2]: https://gpuopen.com/learn/using_matrix_core_amd_rdna4/

<script src="https://utteranc.es/client.js"
        repo="aschrein/aschrein.github.io"
        issue-term="pathname"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>