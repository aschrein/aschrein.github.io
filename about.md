---
layout: page
title: About
permalink: /about/
---

#### Toy projects:
  * [WebGPU MLP image compressor](https://aschrein.github.io/mlp_compression/)
    * Based off of Jakub Boksansky's [d3d12 one](https://github.com/boksajak/Dx12NN/)
  * [Software GPU simulator + Vulkan API(+JIT shader compiler for x86)](https://github.com/aschrein/vulkenstein)
    * Implemented a subset of vulkan API as a virtual device, enough to take some SPIRV pixel/vertex shaders, PSO and rasterize a basic imgui sample and put on screen. Tiny compiler layer takes spirv and use llvm to make a wavefront, linearize CFG and compile to x86 for native execution on cpu.
  * [GPU simulator in Rust](https://github.com/aschrein/guppy) and [web link](https://aschrein.github.io/guppy/)
    * A toy GPU performance simulator with an imaginary ISA that can spit out basic wave traces based on a config.
  * [3D Prototyping Graphics Framework](https://github.com/aschrein/Vulki)
  * [3D Prototyping Graphics Framework 2](https://github.com/aschrein/VulkII)
  * [3D Prototyping Graphics Framework 3](https://github.com/aschrein/dgfx)
  * [Prototype for node based frame graph/shader editor](https://github.com/aschrein/WebThingy) and [web link](https://aschrein.github.io/thingy/)

#### OSS contributions:
  * Adding a WASM i32x4(SIMD4) target to [ISPC](https://github.com/ispc/ispc/commits?author=aschrein)
  * GPUOpen effects

#### Shader toys:
  * [Octahedral facet solid angle](https://www.shadertoy.com/view/tlBXDd)
  * [Run-length compressed models on GPU](https://www.shadertoy.com/view/tlSSWD)
  * [Minimal TAA(kinda wonky)](https://www.shadertoy.com/view/WlSSWc)
  * [Simple ripples](https://www.shadertoy.com/view/wtjSWh)

## Work experience:
#### February 2021 - Now (100% Remote) [AMD](https://www.amd.com/en)
  * Leading an effort on modeling+inference+data generation for spatio temporal ray tracing monte carlo denoising middleware.
    * i8/f8|wmma|Parametric kernel prediction networks|CNN|Unets|Pytorch|Recurrent
  * C++|d3d12|hlsl| Some internal tooling for testing/capture/replay/rapid prototyping/perf analysis.
  * Leading development of GPUOpen code/samples(some of it shipped in a number of AAA games):
    * [Hybrid Stochastic Reflections Sample](https://gpuopen.com/learn/hybrid-reflections/)
    * [Reflections Denoiser](https://github.com/GPUOpen-Effects/FidelityFX-Denoiser/tree/d7dfecbabe7b9523b14e7b067216e06b86e8d189/ffx-reflection-dnsr)
    * Brixelizer
      * [Presented by my colleague at GDC 2023](https://www.youtube.com/watch?v=iY15xhuuHPQ)
      * [Presented by my colleague at GDC 2024](https://www.youtube.com/watch?v=dQ2XtHaPN9w)
      

#### June 2020 - November 2020 (100% Remote) [Unigine](https://unigine.com/)
  * Engine development.
  * Prototyped/Envisioned [Visual programming(node based) system for material generation](https://unigine.com/blog/2020/09/30/feature-preview-shader-graph-editor/)

#### May 2017 - May 2019 (Gdansk/Poland) [Intel](https://www.intel.com/)
  * Worked briefly on CPU compilers(icc/Big endian) and runtime libraries(Embedded).
  * Developed a bespoke gfx API capture/replay tool for experimentation on pass reordering, perf stat gathering and geometry manipulations. Was using internal gpu hw counters api to get high frequency zoomed in picture of the bottlenecks. Was mature enough to capture a bunch of AAA projects.
  * Prototyping graphics compiler features for game workloads.
  * Game performance analysis on current/future platforms.
  * Source-to-Source clang based transpilation tool.

#### Jan 2017 - May 2017 (Novosibirsk/Russia) [Bricsys](https://www.bricsys.com/)
  * Direct modeling tools for BricsCAD. Focused on prototyping a tool to copy paste confined geometric features using topological dual representation.

#### Dec 2015 - Aug 2016 (Novosibirsk/Russia) АО «НТЦ ЕЭС»
  * In-House math and vector graphics libraries.
  * Working on my bachelor's thesis on solving steady state electrical current configs for large power networks.

## Education:
* 2016 BSc in physics at [NSU](https://english.nsu.ru/) (Novosibirsk/Russia)

## Social:

[Twitter](https://twitter.com/antonschrein)

[mastodon](https://mastodon.gamedev.place/@aschrein)

[shadertoy](https://www.shadertoy.com/user/aschrein)

[LinkedIn](https://www.linkedin.com/in/anton-schreiner-b7a375200/)
