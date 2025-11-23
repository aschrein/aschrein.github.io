---
layout: post
title:  "Generating Cats with learned lookup tables"

date:   2025-11-23 01:00:00 +0000
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
This is a follow up post to the [previous one](https://aschrein.github.io/jekyll/update/2025/11/22/generating_cats.html) about generating cats with KPN filtering. Here I explore using lookup tables per 8x8 image tokens using a dictionary of 512/64 8x8 patterns.

# Post

I expected this to not work very well, as the expressivity of the model is limited by the size of the dictionary, but surprisingly it works reasonably well and produces recognizable cat images. Also I thought that the LUT dictionary would learn interpreatable patterns, but it seems to just learn some arbitrary basis.

Each 8x8 patch is a softmax sum over the 512 learned patterns, so the model can interpolate between them. The model is a patch transfomer with 16 stacked self attention blocks operating on 64 tokens per image(8x8 RGB patch is mapped to a token), it outputs logits or weights of the LUT per 8x8 patch. Each LUT entry is a learnable 8x8 RGB patch that is static during inference. It is trained the same way as before, by lerping to noise and predicting the original image. At inference we run the model iteratively starting from Gaussian noise and gradually unlerping to the predicted image.

LUT:

![](/assets/gen_cats2/lut1.png)

Samples:

![](/assets/gen_cats2/samples.png)


Zoomed in:

![](/assets/gen_cats2/zoom1.png)

Actually worked much better than I expected!

After thinking about it a bit more, it makes sense that this works, as the model can learn a dictionary of basis patterns. If we have 8x8 RGB patches, then we have effectively 64x3=192 dimensions per patch. With 512 patterns we more than double the number of basis vectors, so the model is not as limited as I initially thought.

Now the question is how well this would do if we use a limited number of dictionary entries, like 64.
What I will do as well is penalize off-diagonal entries in the Gram matrix of the learned patterns to encourage orthogonality. And on top of that I will use unnormalized tanh weights for the LUT combination, because softmax doesn't really make sense here, as we don't want the result to be a convex combination necessarily.

Here's the Gram matrix:

![](/assets/gen_cats2/gram.png)

LUT:

![](/assets/gen_cats2/lut2.png)

Eearly samples at 3k epochs:

![](/assets/gen_cats2/samples2.png)

Zoomed in:

![](/assets/gen_cats2/zoom2.png)

Later samples at 7k epochs:

![](/assets/gen_cats2/samples3.png)

![](/assets/gen_cats2/zoom3.png)

![](/assets/gen_cats2/zoom4.png)

Not bad at all! The model struggled a bit at first but then improved and started generating consistently good samples, I'm not measuring FID/diversity or anything formal here, just eyeballing it for the signs of life. There's probably problems with this approach, but so far it looks interesting.

[Code](https://github.com/aschrein/pyd3d12/blob/master/tests/torch/cat_diffusion6.py)

The next experiment I wanted to try is to compute the LUT dynamically by making the model to output a set of vectors that I would use to cook it up as RGB outer products, so it would output N * (3 * 2 * 8) and then each color channel tile would be computed as the outer product of the two 8-d vectors. This would potentially increase the capacity of the model as it could generate more diverse patterns on the fly.

I also added learnable static tokens that are appended to the patch embeddings before feeding them to the transformer, this would allow the model to capture some static information that is shared across all images, like common textures or colors.

5k epochs:

![](/assets/gen_cats2/samples4.png)

![](/assets/gen_cats2/zoom5.png)

10k epochs:

![](/assets/gen_cats2/samples5.png)

![](/assets/gen_cats2/zoom6.png)

[Code](https://github.com/aschrein/pyd3d12/blob/master/tests/torch/cat_diffusion7.py)

The results look promising, the model seems to be able to generate decent cat images even with the limited capacity of the LUT approach. The dynamic LUT generation seems to help with diversity and detail.


# Links

<script src="https://utteranc.es/client.js"
        repo="aschrein/aschrein.github.io"
        issue-term="pathname"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>