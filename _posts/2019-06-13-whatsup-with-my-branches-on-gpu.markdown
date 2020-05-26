---
layout: post
title:  "What's up with my branch on GPU?"
date:   2019-06-13 08:19:16 +0000
categories: jekyll update
---
## About
This post is addressed to those who are interested to learn more about how GPU handles branching and targeted as an introduction to the topic. I recommend skimming through \[[1]\], \[[2]\], \[[17]\] and \[[8]\] to get an idea of what GPU execution model looks like in general because here we're gonna take a closer look at one particular detail. A curious reader shall find all references at the end of the post. If you find any mistakes please reach out.

### TL;DR;
GPUs are like slow CPUs with many cores, wide vector units and memory bus. GPUs handle branches the same way vectorized CPU code does: scalarization. Your code is being linearized into a linear stream of vector instructions and masking is used to discard results for disabled lanes. Linearization means runtime is bound by runtime of the slowest lane.
{: .info}


## Table of content
* this unordered seed list will be replaced
{:toc}
## Vocabulary

| GPU | Graphics processing unit |
| SIMD | Single instruction multiple data: Flynn's taxonomy |
| SIMT | Single instruction multiple threads |
| SPMD | Single program multiple data - a model of programming used in shaders/ispc when vectorization is implicit |
| SIMD wave | Thread executing in SIMD mode |
| Lane | designated data stream in SIMD model |
| SMT | Simultaneous multi-threading (Intel Hyper-threading)\[[2]\]* Multiple threads share computational resources of a core |
| IMT | Interleaved multi-threading\[[2]\]* Multiple threads share computational resources of a core but only one executes per clock |
| BB | Basic Block - linear sequence of instructions with only one jump at the end |
| CFG | Control Flow Graph - a graph whose nodes are BB and edges represent jumps |
| ILP | Instruction Level Parallelism\[[3]\] |
| ISA | Instruction Set Architecture |
| SLM | Shared Local Memory, Group Local |


Throughout this post I'm referring to this fictional taxonomy. It approximates how a modern GPU is organized.
```
Hardware:
GPU  -+
      |- core 0 -+
      |          |- wave 0 +
      |          |         |- lane 0
      |          |         |- lane 1
      |          |         |- ...
      |          |         +- lane Q-1
      |          |
      |          |- ...
      |          +- wave M-1
      |            
      |- ...
      +- core N-1

* ALU - SIMD ALU unit for short

Software:
group +
      |- thread 0
      |- ...
      +- thread N-1
```
Also some names:

| core | CU(compute unit), SM(streaming multiprocessor), EU(execution unit)  |
| wave | wavefront, HW thread, warp, context |
| lane | SW thread(called sometimes) |


## What is so special about GPU core compared to CPU core?
Any current generation single GPU core is less beefy compared to what you may encounter in the CPU world: simple ILP/multi-issue\[[6]\] and prefetch\[[5]\], no speculation or branch/return prediction. All of this coupled with tiny caches frees up quite a lot of the die area which gets filled with more cores with wider vector units. Memory load/store machinery is able to handle bandwidths of an order of magnitude larger(not true for integrated/mobile GPUs that share the bus with CPU) than that of a typical CPU at a cost of more latency.

Sometimes latency limits bandwidth. Async memory requests need tracking, tracking needs state, state needs on-die area, on-die area is limited -> async mem requests are a limited resource. When an on-fly memory request queue gets filled the next mem request just blocks execution. That could be a corner case with a long stream of independent memory stores/writes.
{: .note}

GPU employs SMT\[[2]\] to hide latency - while one wave is stalled, another utilizes free computation resources of a core. Because memory latency on GPU is quite high GPUs usually have an explicit big register file. Typically the number of waves handled by one core depends on the number of registers used and determined dynamically by allocating on a fixed register file\[[8]\]. The instruction scheduling is hybrid dynamic and static\[[6]\] \[[11] 4.4\].

SMT cores execute in SIMD mode.  
{: .note}

From a software point of view some high level language is used with implicit vectoriztion model which does not expose all the HW details like SIMD width or instruction scheduling. However, typically kernels get compiled into SIMD instructions where each lane corresponds to one software thread. Therefore the actual hardware thread(wave) executes N software threads in lockstep.  
### Illustrations of GPU core
![Figure 1](/assets/legend.png)  
###### Diagram color-state coding
![Figure 1](/assets/interleaving.png)  
###### Figure 1. Execution trace 4:2
Made on my toy [gpu simulator](https://aschrein.github.io/guppy/)(not user friendly).  
The image shows a trace of an execution mask where the x axis is time from left to right(1 clock per pixel) and the y axis is 1 lane state per pixel and waves in SIMD32 mode from top to bottom.
This is an illustration of how a GPU core Execution trace might look like for a fantasy configuration: four waves share one sampler and two ALU units. Wave scheduler dispatches two instructions from two waves each cycle. When a wave stalls on memory access or long ALU operation, the scheduler switches to another pair of waves making ALU units almost 100% busy all the time.  

Having many threads executing on a single core makes features like speculative execution and branch prediction unnecessary for performance advancement. Because when one thread is blocked, another may start executing. That's why sometimes it's recommended to have graphics+compute workloads on GPU as they're likely to have orthogonal bottlenecks which is a win in the SMT model provided the register file can fit many threads.
{: .note}

![Figure 2](/assets/interleaving_2.png)  
###### Figure 2. Execution trace 4:1
This is the same workload but this time only one wave issues instructions each cycle. Note how the second ALU is starving.  
![Figure 3](/assets/interleaving_3.png)  
###### Figure 3. Execution trace 4:4
This time four instructions are issued each cycle. Note that ALUs are oversubscribed in this case so two waves idle almost all the time(actually it's a pitfall of the scheduling algorithm).  

Read more about scheduling challenges\[[12]\].
{: .note}

Real world GPUs have different configurations per core: some may have up to 40 waves per core and 4 ALUs, some have fixed 7 waves and 2 ALUs. It all depends on a variety of factors and is determined through a thorough architecture simulation process.
Also real SIMD ALUs may have narrower width than those of waves they serve, it then takes multiple cycles to process one issued instruction, the multiplier is called 'chime' length\[[3]\].

## What is coherence/divergence?
Let's look at the following kernel:
###### Example 1
```c++
uint lane_id = get_lane_id();
if (lane_id & 1) {
    // Do smth
}
// Do some more
```
Here we see code where the execution path depends on the id of the lane being executed. Apparently different lanes have different values. So what should happen? There are different approaches to tackle this problem \[[4]\] but eventually they do approximately the same thing. One of such approaches is using execution masks which I will focus on. This approach is employed by pre-Volta Nvidia and AMD GCN GPUs. The core of execution mask is that we keep a bit for each lane within the wave. If a lane has 0 set to its corresponding execution bit no registers will be touched for that lane by the next issued instruction. On source code level it's a bit more involved since the compiler can do tricks but eventually the observed state for a lane should be as if it was just scalar code. Effectively the lane shouldn't feel the impact(again, it's about the observable state) of all the executed instruction as long as it's execution bit is 0. The compiler usually linearizes control flow so that instruction pointer iterates over all instructions for all lanes. In the above example there'd be no branch! it's not needed on vector machine, you just switch lane bits. General GFC linearization is a big and complex topic but it should be straightforward for little code snippets.
So let's say we have waves of width 8. This is how execution mask will look like for the linearized kernel:
###### Example 1. Execution mask history
```c++
                                  // execution mask
uint lane_id = get_lane_id();     // 11111111
if (lane_id & 1) {                // 11111111
    // Do smth                    // 01010101
}
// Do some more                   // 11111111
```

Quick intro into control flow graphs:
```
The above example is           Commonuse languages build so called
equivalent to the              "Structured control flow" with these
following CFG:                 single-entry-single-exit building blocks:
     A                                A       A   A<-+
    /|                               / \     /|   |  |
   B |                              B   C   B |   |  |
    \|                               \ /     \|   |  |
     C                                D       C   B--+

General CFGs may include something like this:
    |   |
    A   B
   / \ / \  
  C   D   E 
  |   |   |
Which we aren't really interested in because our languages and hardware are limited.

The linearization could be thought of as the following transformation:
   A   
  / \  
 B   C  ==> A->B->C->D
  \ /  
   D   

    A 
   /| 
  B | ==> A->B->C
   \| 
    C 

 A<-+
 |  |
 |  | ==> A-->B  Wait, nothing's changed. True. Loops must still be there but they
 |  |     ^   |  are kind o special and are called 'back edges' meaning that linear
 B--+     |   |  flow must go back.
          +---+

```

Look at some more examples and try imagining how it's linearized:
###### Example 2
```c++
uint lane_id = get_lane_id();
for (uint i = lane_id; i < 16; i++) {
    // Do smth
}
```
###### Example 3
```c++
uint lane_id = get_lane_id();
if (lane_id < 16) {
    // Do smth
} else {
    // Do smth else
}
```
Ok that's enough.
###### Example 4
```c++
uint var = get_lane_id();
if (var < 16) {
    while (true) {
        if (foo(var) == 0) {
            while (true) {
                if (bar(var))
                    break;
                if (oops(var))
                    goto changed_my_mind;
            }
            break;
        } else {
            continue;
        }
        if (fun(var) == 666) {
            var = fen(var);
        }
    }
} else {
    while (true) {
changed_my_mind:
        var = boom(var);
        if (var == 0)
            break;
    }
}
```
I said that's enough!

The last example won't work on most GPUs because of the goto. Not because of the goto per se but due to its target. It creates a loop with many entries. This is called irreducible control flow. In rigid terms reducible control flow graph is when you can split edged in two groups: one forms a connected DAG(simple if/else), the other one has edges where target dominates source(structured loops). 

Dominator of A is such a B where A and B are nodes of the same connected component and every execution must come through B to reach A. In other words any variable declared at B is visible at A.
{: .note}

The above example could work if the compiler can make proper transformations to get rid of the nasty loop.
{: .note}

With the execution mask approach usually some kind of stack is employed by the HW to be able to traverse complicated graphs. A naive approach is to keep a stack of tuples (exec_mask, address) and add reconvergence instructions that pop a mask from the stack and change the instruction pointer for the wave. In that way a wave will have enough information to traverse the whole CFG for each lane.  
From the performance point of view, it takes a couple of cycles just to process a control flow instruction because of all the bookkeeping. And don't forget that the stack has limited depth.  
{: .note}

By courtesy of [@craigkolb](https://twitter.com/craigkolb) I've read \[[13]\] in which it is noted that AMD GCN fork/join instructions select the path with the fewer number of threads first \[[11]4.6\] which guarantees that log2 depth of the mask stack is enough.
{: .note}  

Apparently it's almost always possible to inline everything/structurize CFGs in a shader and therefore keep all execution mask history in registers and schedule CFG traversal/reconvergence statically\[[15]\]. Skimming through LLVM backend for AMDGPU I didn't find any evidence of stack handling ever being emitted by the compiler.  
{: .note}

### HW support for execution mask
Now take a look at these control flow graphs(image from Wikipedia):  
![Figure 4](/assets/Some_types_of_control_flow_graphs.png)  
###### Figure 4. Some types of control flow graphs
So what is the minimal set of mask control instructions we need to handle all cases? Here is what it looks like in my toy ISA with implicit parallelization, explicit mask control and fully dynamic data hazard synchronization:
```nasm
push_mask BRANCH_END         ; Push current mask and reconvergence pointer
pop_mask                     ; Pop mask and jump to reconvergence instruction
mask_nz r0.x                 ; Set execution bit, pop mask if all bits are zero

; Branch instruction is more complicated
; Push current mask for reconvergence
; Push mask for (r0.x == 0) for else block, if any lane takes the path
; Set mask with (r0.x != 0), fallback to else in case no bit is 1
br_push r0.x, ELSE, CONVERGE 
```
Let's take a look at what d) case might look like.

```nasm
A:
    br_push r0.x, C, D
B:
C:
    mask_nz r0.y
    jmp B
D:
    ret
```

Read more on GCN support for control flow instructions \[[11]\] ch.4 and LLVM implementation \[[15]\].
{: .note}

Bottom line:


`Divergence` - Emerging difference in execution paths taken by different lanes of the same wave.
{: .warning}

Hardware needs extra instructions/registers/stack to handle execution mask. 
{: .warning}

More branches - more register pressure as more values have overlapping life spans.
{: .warning}

Control flow graph is linearized.  
{: .warning}

## Execution mask handling examples
This is not essential material so it's collapsed to save up some space.

{::options parse_block_html="true" /}
<details><summary class="info" markdown="span">Click to show!</summary>

### Fictional ISA
I compiled the previous code snippets into my toy ISA and ran it on the simulator at SIMD32. Take a look at how it handles execution masks.  
***Update*** Note that the toy simulator always selects the true path first which is not the best method.
###### Example 1
```nasm
; uint lane_id = get_lane_id();
    mov r0.x, lane_id
; if (lane_id & 1) {
    push_mask BRANCH_END
    and r0.y, r0.x, u(1)
    mask_nz r0.y
LOOP_BEGIN:
    ; // Do smth
    pop_mask                ; pop mask and reconverge
BRANCH_END:
    ; // Do some more
    ret
```
![Figure 5](/assets/branch_1.png)
###### Figure 5. Example 1 Execution trace
Did you Notice the black area? It is wasted time. Some lanes are waiting for others to finish iterating.
###### Example 2
```nasm
; uint lane_id = get_lane_id();
    mov r0.x, lane_id
; for (uint i = lane_id; i < 16; i++) {
    push_mask LOOP_END        ; Push the current mask and the pointer to reconvergence instruction
LOOP_PROLOG:
    lt.u32 r0.y, r0.x, u(16)  ; r0.y <- r0.x < 16
    add.u32 r0.x, r0.x, u(1)  ; r0.x <- r0.x + 1
    mask_nz r0.y              ; exec bit <- r0.y != 0 - when all bits are zero next mask is popped
LOOP_BEGIN:
    ; // Do smth
    jmp LOOP_PROLOG
LOOP_END:
    ; // }
    ret
```
![Figure 6](/assets/branch_2.png)
###### Figure 6. Example 2 Execution trace
###### Example 3
```nasm
    mov r0.x, lane_id
    lt.u32 r0.y, r0.x, u(16)
    ; if (lane_id < 16) {
        ; Push (current mask, CONVERGE) and (else mask, ELSE)
        ; Also set current execution bit to r0.y != 0
    br_push r0.y, ELSE, CONVERGE
THEN:
    ; // Do smth
    pop_mask
    ; } else {
ELSE:
    ; // Do smth else
    pop_mask
    ; }
CONVERGE:
    ret
``` 
![Figure 7](/assets/branch_3.png)
###### Figure 7. Example 3 Execution trace

### AMD GCN ISA
***Update*** GCN also uses an explicit mask handling, you can read more about it here\[[11] 4.x\]. I decided it's worth putting some examples with their ISA, thanks to [shader-playground](http://shader-playground.timjones.io).  
Note that the compiler is smart, you may get a different result. I tried to fool the compiler into not optimizing my branches by putting pointer chase loops in there then cleaned up the assembly.  
Also note that S_CBRANCH_I/G_FORK and S_CBRANCH_JOIN instructions are not used in these snippets due to their simplicity/lack of compiler support. Therefore unfortunately the mask stack is not covered. If you know how to make the compiler spit stack handling please convey this information.  
***Update*** Watch this [talk](https://youtu.be/8K8ClHoZzHw) by [@SiNGUL4RiTY](https://twitter.com/SiNGUL4RiTY) about the implementation of vectorized control flow in LLVM backend employed by AMD.  
###### Example 1
```nasm
; uint lane_id = get_lane_id();
; GCN uses 64 wave width, so lane_id = thread_id & 63
; There are scalar s* and vector v* registers
; Execution mask does not affect scalar or branch instructions
    v_mov_b32     v1, 0x00000400      ; 1024 - group size
    v_mad_u32_u24  v0, s12, v1, v0    ; thread_id calculation
    v_and_b32     v1, 63, v0
; if (lane_id & 1) {
    v_and_b32     v2, 1, v0
    s_mov_b64     s[0:1], exec        ; Save the execution mask
    v_cmpx_ne_u32  exec, v2, 0        ; Set the execution bit
    s_cbranch_execz  ELSE             ; Jmp if all exec bits are zero
; // Do smth
ELSE:
; }
; // Do some more
    s_mov_b64     exec, s[0:1]        ; Restore the execution mask
    s_endpgm
```
###### Example 2
```nasm
; uint lane_id = get_lane_id();
    v_mov_b32     v1, 0x00000400
    v_mad_u32_u24  v0, s8, v1, v0     ; Not sure why s8 this time and not s12
    v_and_b32     v1, 63, v0
; LOOP PROLOG
    s_mov_b64     s[0:1], exec        ; Save the execution mask
    v_mov_b32     v2, v1
    v_cmp_le_u32  vcc, 16, v1
    s_andn2_b64   exec, exec, vcc     ; Set the execution bit
    s_cbranch_execz  LOOP_END         ; Jmp if all exec bits are zero
; for (uint i = lane_id; i < 16; i++) {
LOOP_BEGIN:
    ; // Do smth
    v_add_u32     v2, 1, v2
    v_cmp_le_u32  vcc, 16, v2
    s_andn2_b64   exec, exec, vcc     ; Mask out lanes which are beyond loop limit
    s_cbranch_execnz  LOOP_BEGIN      ; Jmp if non zero exec mask
LOOP_END:
    ; // }
    s_mov_b64     exec, s[0:1]        ; Restore the execution mask
    s_endpgm
```
###### Example 3
```nasm
; uint lane_id = get_lane_id();
    v_mov_b32     v1, 0x00000400
    v_mad_u32_u24  v0, s12, v1, v0
    v_and_b32     v1, 63, v0
    v_and_b32     v2, 1, v0
    s_mov_b64     s[0:1], exec        ; Save the execution mask
; if (lane_id < 16) {
    v_cmpx_lt_u32  exec, v1, 16       ; Set the execution bit
    s_cbranch_execz  ELSE             ; Jmp if all exec bits are zero
; // Do smth
; } else {
ELSE:
    s_andn2_b64   exec, s[0:1], exec  ; Inverse the mask and & with previous
    s_cbranch_execz  CONVERGE         ; Jmp if all exec bits are zero
; // Do smth else
; }
CONVERGE:
    s_mov_b64     exec, s[0:1]        ; Restore the execution mask
; // Do some more
    s_endpgm
``` 
### AVX512
***Update*** [@tom_forsyth](https://twitter.com/tom_forsyth) pointed out that AVX512 extension comes with an explicit mask handling too, so here are some examples. You can read more about it at \[[14]\] par. 15.x and 15.6.1. It's not precisely a GPU but still a legit SIMD16 at 32 bit. Snippets are made using [godbolt's](https://godbolt.org/z/kwrr1y) ISPC(--target=avx512knl-i32x16) and tampered with heavily.
###### Example 1
```nasm
    ; Imagine zmm0 contains 16 lane_ids
    ; AVXZ512 comes with k0-k7 mask registers
    ; Usage:
    ; op reg1 {k[7:0]}, reg2, reg3
    ; k0 can not be used as a predicate operand, only k1-k7
; if (lane_id & 1) {
    vpslld       zmm0 {k1}, zmm0, 31  ; zmm0[i] = zmm0[i] << 31
    kmovw        eax, k1              ; Save the execution mask
    vptestmd     k1 {k1}, zmm0, zmm0  ; k1[i] = zmm0[i] != 0
    kortestw     k1, k1
    je           ELSE                 ; Jmp if all exec bits are zero
; // Do smth
    ; Now k1 contains the execution mask
    ; We can use it like this:
    ; vmovdqa32 zmm1 {k1}, zmm0
ELSE:
; }
    kmovw        k1, eax              ; Restore the execution mask
; // Do some more
    ret
```
###### Example 2
```nasm
 ; Imagine zmm0 contains 16 lane_ids
    kmovw         eax, k1               ; Save the execution mask
    vpcmpltud     k1 {k1}, zmm0, 16     ; k1[i] = zmm0[i] < 16
    kortestw      k1, k1
    je            LOOP_END              ; Jmp if all exec bits are zero
    vpternlogd    zmm1 {k1}, zmm1, zmm1, 255   ; zmm1[i] = -1
; for (uint i = lane_id; i < 16; i++) {
LOOP_BEGIN:
; // Do smth
    vpsubd        zmm0 {k1}, zmm0, zmm1 ; zmm0[i] = zmm0[i] + 1
    vpcmpltud     k1 {k1}, zmm0, 16     ; masked k1[i] = zmm0[i] < 16
    kortestw      k1, k1
    jne           LOOP_BEGIN            ; Break if all exec bits are zero
LOOP_END:
; // }
    kmovw        k1, eax                ; Restore the execution mask
; // Do some more
    ret
```
###### Example 3
```nasm
 ; Imagine zmm0 contains 16 lane_ids
; if (lane_id & 1) {
    vpslld       zmm0 {k1}, zmm0, 31  ; zmm0[i] = zmm0[i] << 31
    kmovw        eax, k1              ; Save the execution mask
    vptestmd     k1 {k1}, zmm0, zmm0  ; k1[i] = zmm0[i] != 0
    kortestw     k1, k1
    je           ELSE                 ; Jmp if all exec bits are zero
THEN:
; // Do smth
; } else {
ELSE:
    kmovw        ebx, k1
    andn         ebx, eax, ebx
    kmovw        k1, ebx              ; mask = ~mask & old_mask
    kortestw     k1, k1
    je           CONVERGE             ; Jmp if all exec bits are zero
; // Do smth else
; }
CONVERGE:
kmovw            k1, eax              ; Restore the execution mask
; // Do some more
    ret
```

</details>
{::options parse_block_html="false" /}

## How to fight divergence?
I tried to come up with a simple yet complete illustration for the inefficiency introduced by combining divergent lanes.  
Imagine a simple kernel like this:  
```c++
uint thread_id = get_thread_id();
uint iter_count = memory[thread_id];
for (uint i = 0; i < iter_count; i++) {
    // Do smth
}
```
Let's spawn 256 threads and measure the duration:  
![Figure 8](/assets/rand.png)  
###### Figure 8. Divergent threads execution time
The x axis is SW thread id, the y axis is clock cycles; the different bars show how much time is wasted by grouping threads with different wave widths compared to single threaded execution.  
The execution time of a wave is equal to the maximum execution time among confined lanes. You can see that the performance is already ruined at SIMD8, further widening just makes it slightly worse.  
![Figure 9](/assets/sorted.png)
###### Figure 9. Coherent threads execution time  
This figure shows the same bars but this time iteration counts are sorted over thread ids, so that threads with similar iteration counts get dispatched to the same wave.  
For this example the potential speedup is around 2x.  

Bottom line:

Sort input data
{: .warning}

For example, if you are writing a ray tracer, grouping rays with similar direction and position could be beneficial because they are likely to be traversing the same nodes in BVH. For more details please follow \[[10]\] and related articles.
{: .note}

Keep CFG simple
{: .warning}

Complex CFG are not a problem per se they're just more likely to cause divergent execution. Really your only problem is divergence, not the complexity of the CFG.
{: .note}

## Divergent memory access
Generally speaking every SIMD load is a gather and every SIMD store is a scatter. A memory operation is generated for each active lane when a store/load instruction is issued, typically if inactive lanes have invalid addresses at the corresponding address slots, no exception is going to be generated.  
Memory coalescing machinery is going to take care of optimizing apparent patterns to the global memory which is one of the benefits of gather loads. In case you access SLM you may hit a bank collision issue.  
{: .note}

### Sample in a branch
Things indeed might get hairy if you are trying to sample a texture in a branch. Particularly, if you are sampling in a pixel shader and use anisotropic/trilinear filtering - those kinds of features depend on HW gradients for implicit LOD selection which require that all lanes participating in a 2x2 pixel group have valid arguments. The way it works is that HW packs adjacent pixel groups of 2x2 in the same wave. This has a consequence that 1 pixel triangle is going to spawn 4 pixels and at least 1 wave, the same happens with lone pixels on triangle boundaries. Invisible pixels are called helpers. Some helper pixels are going to have their barycentric coordinates outside the triangle.  
A good read on the subject matter is [DirectX-Specs 16.8.2 Restrictions on Derivative Calculations](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#16.8%20Interaction%20of%20Varying%20Flow%20Control%20With%20Screen%20Derivatives).

In short the spec allows such samples if the texture address is a shader input or a statically indexed constant. It makes sense because even if the HW does not know how to handle divergent samples, its compiler can just hoist that sample from the branch and eliminate the issue completely. I expect other APIs to enforce similar behaviour.  

[@Themaister](https://twitter.com/Themaister) Wrote a post \[[18]\] about his experiments with texture samples in a divergent control flow on different HW. 
{: .note}

The other issue arises when different lanes try to sample different textures. This is quite usual nowadays with deferred everything and bindless resources.
On vulkan there's the nonuniformEXT compiler hint that the programmer must put when divergent indexing could take place.

```c++
// Note nonuniformEXT
color = texture(sampler2D(diffuseMaps[nonuniformEXT(materialID)], filter), uv);
```
On AMD an image descriptor is a scalar value that takes up to 8 registers. Therefore the divergent indexing must be scalarized.  
On AMD nonuniformEXT is equivalent(more or less) to this loop:

```c++
while (true) {
    uint currentIdx = readFirstInvocationARB(materialID);  // Get textureIdx for the lowest _active_ lane
    if (currentIdx == materialID) { // Lanes with the same index enter the branch
        color = texture(sampler2D(diffuseMaps[currentIdx], filter), uv); // And sample the texture
        break; // Disable lanes that finished sampling
    } // Other _active_ lanes continue looping
}
```
Which is compiled to this assembly:

![](/assets/nonuniform_sample_auto.png)

Note `image_sample v[0:2], v[2:4], s[16:23], s[4:7]` in a loop. In this case `s[16:23]` holds the image descriptor.
{: .note}

## Avoid branching at all cost?
No. It's impossible. Just pay attention to divergence and profile. Branches are a huge win when used properly.

## Extra
It's worth mentioning that there are some techniques to grapple with divergence on HW level, some of them are Dynamic Warp Formation\[[7]\] and predicated execution for small branches.

One of the advanced techniques for dynamic work balancing is called ‘persistent threads’ read more at \[[19]\]. The gist of this method is that you create just enough threads to saturate your GPU with a generic kernel which dynamically grabs job items off a global queue using an atomic counter. With coherent memory like the one you could have with CUDA(Unified memory in NV terminology) it becomes even more interesting because you can stream work from CPU to GPU online.

The amount of in-flight memory requests being served is somehow limited by the HW. So if you have a kernel with tons of samples or memory loads, it might stall just trying to issue those requests. Which means that some amount of interleaving of ALU instructions and memory requests is needed to avoid such stalls. But fear not, it's usually taken care of in the compiler, fortunately shader memory models are quite permissive when it comes to reordering.  

# Links

[1][A trip through the Graphics Pipeline][1]

[1]: https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/

[2][Kayvon Fatahalian: PARALLEL COMPUTING][2]

[2]: http://cs149.stanford.edu/winter19/

[3][Computer Architecture A Quantitative Approach][3]

[3]: https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1

[4][Stack-less SIMT reconvergence at low cost][4]

[4]: https://hal.archives-ouvertes.fr/hal-00622654/document

[5][Dissecting GPU memory hierarchy through microbenchmarking][5]

[5]: https://arxiv.org/pdf/1509.02308&ved=0ahUKEwifl_P9rt7LAhXBVxoKHRsxDIYQFgg_MAk&usg=AFQjCNGchkZRzkueGqHEz78QnmcIVCSXvg&sig2=IdzxfrzQgNv8yq7e1mkeVg

[6][Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking][6]

[6]: https://arxiv.org/pdf/1804.06826.pdf

[7][Dynamic Warp Formation and Scheduling for Efficient GPU Control Flow][7]

[7]: http://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/fung07_dynamicwarp.pdf

[8][Maurizio Cerrato: GPU Architectures][8]

[8]: https://t.co/QG28evt7QR

[9][Toy GPU simulator][9]

[9]: https://aschrein.github.io/guppy/

[10][Reducing Branch Divergence in GPU Programs][10]

[10]: https://www.eecis.udel.edu/~cavazos/cisc879-spring2012/papers/a3-han.pdf

[11]["Vega" Instruction Set Architecture][11]

[11]: https://developer.amd.com/wp-content/resources/Vega_Shader_ISA_28July2017.pdf

[12][Joshua Barczak:Simulating Shader Execution for GCN][12]

[12]: http://www.joshbarczak.com/blog/?p=823#

[13][Tangent Vector: A Digression on Divergence][13]

[13]: https://tangentvector.wordpress.com/2013/04/12/a-digression-on-divergence/

[14][Intel® 64 and IA-32 ArchitecturesSoftware Developer’s Manual][14]

[14]: https://software.intel.com/sites/default/files/managed/39/c5/325462-sdm-vol-1-2abcd-3abcd.pdf

[15][Vectorizing Divergent Control-Flow for SIMD Applications][15]

[15]: https://github.com/rAzoR8/EuroLLVM19

[16][Jason Booth: Terrain Shader Generation Systems][16]

[16]: https://80.lv/articles/using-next-gen-terrain-engines-for-games-production-009snw/?fbclid=IwAR3L7HgMjz8aCCIy_XQc-Tedn7JOHrYzIR10QvZRTEbOCcLFWu6izc7bS4M

[17][Matthäus G. Chajdas: Introduction to compute shaders][17]

[17]: https://anteru.net/blog/2018/intro-to-compute-shaders/

[18][Maister's Graphics Adventures: The weird world of shader divergence and LOD][17]

[18]: http://themaister.net/blog/2019/09/12/the-weird-world-of-shader-divergence-and-lod/

[19][A Specialized Concurrent Queue for Scheduling Irregular Workloads on GPUs][19]

[19]: https://www.researchgate.net/publication/334691952_A_Specialized_Concurrent_Queue_for_Scheduling_Irregular_Workloads_on_GPUs

[20][Rendering of Surge][20]

[20]: https://www.slideshare.net/philiphammer/dissecting-the-rendering-of-the-surge

[21][Wave programming with Vulkan][21]

[21]: http://32ipi028l5q82yhj72224m8j.wpengine.netdna-cdn.com/wp-content/uploads/2017/07/GDC2017-Wave-Programming-D3D12-Vulkan.pdf

[22][Id Tech6][22]

[22]: http://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf

[23][From Source to ISA: A Trip Down the Shader Compiler Pipeline][23]

[23]: https://www.youtube.com/watch?v=_ilAL-1-moA

[24][INTRO TO GPU SCALARIZATION][24]

[24]: https://flashypixels.wordpress.com/2018/11/