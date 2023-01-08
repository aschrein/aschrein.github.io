---
layout: post
title:  "Using Differentiable Strokes For Image Approximation"
date:   2022-01-08 00:00:00 +0000
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

# Post

Another short post. This time I extended the fourier shape with the angle falloff and made it to concentrate around the rim with exp(-((x - shape.center) - shape.radius)) with learnable width and length. I quite like the results so far so will be moving to the next step in my ML journey.

[notebook](/assets/ml_image_lines/experiment.html)

<video height="256" autoplay loop muted>
<source src="/assets/ml_image_lines/experiment_video_7.mp4" type="video/mp4">
</video>

# Links

[1][geometric-art-with-pytorch][1]

[1]: https://towardsdatascience.com/geometric-art-with-pytorch-c6d92bf3e320/

[2][geometric-art-with-pytorch/youtube][2]

[2]: https://www.youtube.com/watch?v=OSA5fZZwEW4/

[3][https://www.berkayantmen.com/rank.html][3]

[3]: https://www.berkayantmen.com/rank.html/

[4][https://arxiv.org/pdf/2006.16038.pdf][4]

[4]: https://arxiv.org/pdf/2006.16038.pdf/

[4][Photo used for the input][4]

[4]: https://www.pexels.com/photo/curly-hair-woman-wearing-headscarf-14296202/



<script src="https://utteranc.es/client.js"
        repo="aschrein/aschrein.github.io"
        issue-term="pathname"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>