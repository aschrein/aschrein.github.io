---
layout: post
title:  "What's up with my branch on GPU?"
date:   2019-06-13 08:19:16 +0000
categories: jekyll update
---

## About
This post is a small writeup addressed to programmers who are interested to learn more about how GPU handles branching and targeted as an introduction to the topic. I recommend skimming through \[[1]\], \[[2]\] and \[[8]\] to get an idea of what GPU execution model looks like in general because here we're gonna take a closer look at one particular detail. A curious reader shall find all references at the end of the post. If you find any mistakes please reach out.

## Table of content
* this unordered seed list will be replaced
{:toc}
## Vocabulary
* GPU - Graphics processing unit
* Flynn's taxonomy
  * SIMD - Single instruction multiple data
  * SIMT - Single instruction multiple threads
* SIMD wave - Thread executing in SIMD mode
* Lane - designated data stream in SIMD model
* SMT - Simultaneous multi-threading (Intel Hyper-threading)\[[2]\]
  * Multiple threads share computational resources of a core
* IMT - Interleaved multi-threading\[[2]\]
  * Multiple threads share computational resources of a core but only one executes per clock
* BB - Basic Block - linear sequence of instructions with only one jump at the end
* ILP - Instruction Level Parallelism\[[3]\]
* ISA - Instruction Set Architecture

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
Other names:
* core could be CU, SM, EU
* wave could be wavefront, HW thread, warp, context
* lane could be SW thread

## What is so special about GPU core compared to CPU core?
Any current generation single GPU core is less beefy compared to what you may encounter in CPU world: simple ILP/multi-issue\[[6]\] and prefetch\[[5]\], no speculation or branch/return prediction. All of this coupled with tiny caches frees up quite a lot of the die area which gets filled with more cores. Memory load/store machinery is able to handle bandwidths of an order of magnitude larger(not true for integrated/mobile GPUs) than that of a typical CPU at a cost of more latency. GPU employs SMT\[[2]\] to hide this latency - while one wave is stalled, another utilizes free computation resources of a core. Typically the number of waves handled by one core depends on registers used and determined dynamically by allocating on a fixed register file\[[8]\]. The instruction scheduling is hybrid dynamic and static\[[6]\] \[[11] 4.4\]. SMT cores execute in SIMD mode yielding high number of FLOPS.
![Figure 1](/assets/legend.png)  
###### Diagram legend
![Figure 1](/assets/interleaving.png)  
###### Figure 1. Execution history 4:2
The image shows history of execution mask where the x axis is time from left to right and the y axis is lane id from top to bottom. If it does not make sense to you, please return to it after reading the next sections.  
This is an illustration of how a GPU core execution history might look like for a fictional configuration: four waves share one sampler and two ALU units. Wave scheduler dispatches two instructions from two waves each cycle. When a wave stalls on memory access or long ALU operation, scheduler switches to another pair of waves making ALU units almost 100% busy all the time.  
![Figure 2](/assets/interleaving_2.png)  
###### Figure 2. Execution history 4:1
This is the same workload but this time only one wave issues instructions each cycle. Note how the second ALU is starving.  
![Figure 3](/assets/interleaving_3.png)  
###### Figure 3. Execution history 4:4
This time four instructions are issued each cycle. Note that ALUs are oversubscribed in this case so two waves idle almost all the time(actually it's a pitfall of the scheduling algorithm).  

Real world GPUs have different configurations per core: some may have up to 40 waves per core and 4 ALUs, some have fixed 7 waves and 2 ALUs. It all depends on a variety of factors and is determined through thorough architecture simulation process.
Also real SIMD ALUs may have narrower width than those of waves they serve, it then takes multiple cycles to process one issued instruction, the multiplier is called 'chime' length\[[3]\].

## What is coherence/divergence?
Lets look at the following kernel:
###### Example 1
```c++
uint lane_id = get_lane_id();
if (lane_id & 1) {
    // Do smth
}
// Do some more
```
Here we see instruction stream where execution path depends on the id of the lane being executed. Apparently different lanes have different values. So what should happen? There are different approaches to tackle this problem \[[4]\] but eventually they do approximately the same thing. One of such approaches is execution mask which I will focus on. This approach is employed by pre-Volta Nvidia GPUs. The core of execution mask is that we keep a bit for each lane within wave. If a lane has 0 set to its corresponding execution bit no registers will be touched for that lane by the next issued instruction. Effectively the lane shouldn't feel the impact of all the executed instruction as long as it's execution bit is 0. The way it works is that a wave traverses control flow graph in depth first order keeping a history of branches taken. I think it's better to follow an example.  
So lets say we have waves of width 8. This is how execution mask will look like for the kernel:
###### Example 1. Execution mask history
```c++
                                  // execution mask
uint lane_id = get_lane_id();     // 11111111
if (lane_id & 1) {                // 11111111
    // Do smth                    // 01010101
}
// Do some more                   // 11111111
```
Now, take a look at more complicated examples:
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
You'll notice that history is needed. With execution mask approach usually some kind of stack is employed by the HW. A naive approach is to keep a stack of tuples (exec_mask, address) and add reconvergence instructions that pop a mask from the stack and change the instruction pointer for the wave. In that way a wave will have enough information to traverse the whole CFG for each lane.  
From performance point of view, it takes a couple of cycles just to process a control flow instruction because of all the bookkeeping. And don't forget that the stack has limited depth.  
Now take a look at these control flow graphs(image from Wikipedia):  
![Figure 4](/assets/Some_types_of_control_flow_graphs.png)  
###### Figure 4. Some types of control flow graphs
So what is the minimal set of mask control instructions we need to handle all cases? Here is how it looks in my toy ISA with implicit parallelization, explicit mask control and fully dynamic data hazard synchronization:
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
Lets take a look at how d) case might look like.

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
I'm not an expert in control flow analysis or ISA design so I'm sure there is a case that could not be tamed with my toy ISA, although it does not matter as structured CFG should be enough for everyone.

Bottom line:  
* Divergence - emerging difference in execution paths taken by different lanes of the same wave
* Coherence - lack of divergence :)

## Execution mask handling examples
I compiled the previous code snippets into my toy ISA and run it with my simulator which produces cool pictures. Take a look at how it handles execution mask.
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
     mov r0.x, r0.x
     mov r0.x, r0.x
     mov r0.x, r0.x
     mov r0.x, r0.x
     mov r0.x, r0.x
     
    pop_mask                ; pop mask and reconverge
BRANCH_END:
    ; // Do some more
    ret
```
![Figure 5](/assets/branch_1.png)  
###### Figure 5. Example 1 execution history
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
###### Figure 6. Example 2 execution history
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
    mov r0.x, r0.x
    mov r0.x, r0.x
    mov r0.x, r0.x
    mov r0.x, r0.x
    mov r0.x, r0.x

    pop_mask

    ; } else {
ELSE:
    ; // Do smth else
    mov r0.x, r0.x
    mov r0.x, r0.x
    mov r0.x, r0.x
    mov r0.x, r0.x
    mov r0.x, r0.x

    pop_mask
    ; }
CONVERGE:
    ret
``` 
![Figure 7](/assets/branch_3.png)
###### Figure 7. Example 3 execution history

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

Of course the example is too simple but I hope you get the idea: execution divergence stems from data divergence, so keep your CFG simple and data coherent.  
For example, if you are writing a ray tracer, grouping rays with similar direction and position could be beneficial because they are likely to be traversing the same nodes in BVH. For more details please follow \[[10]\] and related articles.

It's worth mentioning that there are some techniques to grapple with divergence on HW level, some of them are Dynamic Warp Formation\[[7]\] and predicated execution for small branches.

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



