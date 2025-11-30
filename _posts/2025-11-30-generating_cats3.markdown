---
layout: post
title:  "Generating Cats using binned gaussian splats"

date:   2025-11-30 01:00:00 +0000
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

## Table of contents
* this unordered seed list will be replaced
{:toc}

# About
This is a follow up post to the [previous one](https://aschrein.github.io/jekyll/update/2025/11/23/generating_cats2.html). This time I explore using binned gaussian splats for generating 8x8 image patches instead of learned lookup tables. 

# Post

The splat kernel is not precisely a gaussian, but a cosine modulated gaussian-like/sinc-like function that oscillates and goes below zero, which helps reduce blur - sort of a wavelet. The function is defined as:

![](/assets/gen_cats3/fn.png)

The idea is that a vision transformer consumes 8x8 patches as tokens, and for each patch it outputs parameters for a mixture of 16 splats. Each splat has a mean position within the patch, a RGB color, a precision matrix defining its shape, and a depth value that controls its blending order. The final patch is rendered by evaluating the splat functions at each pixel and summing them up.

Since the splats are differentiable with respect to their parameters, we can train the model end-to-end using the same lerp-to-noise objective as before. The model learns to output splat parameters that reconstruct the original image patches from noisy inputs.

This acts as a denoising target that we un-lerp iteratively from noise, similar to the previous posts. The model architecture is the same as before, a patch transformer with 16 self-attention blocks operating on 64 tokens per image.

Here are some rollouts:

![](/assets/gen_cats3/samples_0.png)

![](/assets/gen_cats3/samples_1.png)

![](/assets/gen_cats3/samples_2.png)

Actually quite surprising that this trains well, and doesn't hallucinate too much.

The advantage of this approach is that it can be rendered at arbitrary resolutions by simply scaling up the final pixel grid used for rendering of the gaussians, which is not possible with the learned LUTs. This could be useful for generating high-resolution images.

![](/assets/gen_cats3/zoom_0.png)

The problem with this approach is that the model has to learn to output reasonable splat mixture for each patch separately. As a result it has visible seams on the patch boundaries, especially when rendering at higher resolution. Increasing the number of splats per patch or training with longer image size might help, but I didn't explore that further.

Here's an example rendered with splats normalized to be uniform blobs:

![](/assets/gen_cats3/scaled.png)

As we can see the model tries to make the distribution such that splats match on the boundaries with one splat dominating at the center.

[Code](https://github.com/aschrein/pyd3d12/blob/master/tests/torch/cat_diffusion8.py)

# Links

<script src="https://utteranc.es/client.js"
        repo="aschrein/aschrein.github.io"
        issue-term="pathname"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>