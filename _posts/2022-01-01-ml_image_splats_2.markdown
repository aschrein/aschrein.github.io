---
layout: post
title:  "Improving Differentiable Shapes With Depth"
date:   2022-01-01 00:00:00 +0000
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

<video height="256" autoplay loop muted>
<source src="/assets/ml_image_splats/experiment_video_1.mp4" type="video/mp4">
</video>

Differentiable shapes approximate an image.

<video height="256" autoplay loop muted>
<source src="/assets/ml_image_splats_2/experiment_video_1.mp4" type="video/mp4">
</video>

Improved differentiable shapes with the depth function via priority vector.

## Table of content
* this unordered seed list will be replaced
{:toc}

# Post

In the last post I reproduced the results from the differential shapes articles but omitted the depth function. After looking into it I realized it could be done much easier than in the original article without the double argsort().argsort() trick, with just a weight vector.

[notebook](/assets/ml_image_splats_2/experiment.html)  

Per pixel you calculate the weighted average:

```c++
color[x][y] = sum(i=(0, N), shape_color[i] * exp(-shape_w[i]) * shape_f[i][x][y]) / sum(i=(0, N), exp(-shape_w[i]) * shape_f[i][x][y])
```

Where shape_f is the splat function of the ith shape and shape_w is the weight. The weight should go through the exponent to enable the smoothmin kind of effect where you add stuff a * exp(w_a) + b * exp(w_b) and exponent just moves something linearly in digit space. Like if you add numbers in binary, exponentiation is right shift so it moves the bits of one term across the bits of the other term. This has the effect of making the sum depend more on one of the numbers. After the two numbers have been added, the result is normalized to ensure that the smaller number has a lesser impact on the final result.. 0b0001'0000 + 0b0000'0001 = 0b0001'0001, and then normalization makes sure the bits of the smaller term are moved right past the significance.

Also removed the negative color thing, with the weight vector it can learn to put a dark spot on top, which looks much cooler.

I like the sketchy look but not quite satisfied with the overall quality and will think about improving something else.

![png](/assets/ml_image_splats_2/example_input_2.png){: width="256" }

<video height="256" autoplay loop muted>
<source src="/assets/ml_image_splats_2/experiment_video_2.mp4" type="video/mp4">
</video>

![png](/assets/ml_image_splats_2/example_input_3.png){: width="256" }

<video height="256" autoplay loop muted>
<source src="/assets/ml_image_splats_2/experiment_video_3.mp4" type="video/mp4">
</video>


![png](/assets/ml_image_splats_2/example_input_4.png){: width="256" }

<video height="256" autoplay loop muted>
<source src="/assets/ml_image_splats_2/experiment_video_4.mp4" type="video/mp4">
</video>


![png](/assets/ml_image_splats_2/example_input_5.png){: width="256" }

<video height="256" autoplay loop muted>
<source src="/assets/ml_image_splats_2/experiment_video_5.mp4" type="video/mp4">
</video>

![png](/assets/ml_image_splats_2/example_input_6.png){: width="256" }

<video height="256" autoplay loop muted>
<source src="/assets/ml_image_splats_2/experiment_video_6.mp4" type="video/mp4">
</video>

# Links

[1][geometric-art-with-pytorch][1]

[1]: https://towardsdatascience.com/geometric-art-with-pytorch-c6d92bf3e320/

[2][geometric-art-with-pytorch/youtube][2]

[2]: https://www.youtube.com/watch?v=OSA5fZZwEW4/

<script src="https://utteranc.es/client.js"
        repo="aschrein/aschrein.github.io"
        issue-term="pathname"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>