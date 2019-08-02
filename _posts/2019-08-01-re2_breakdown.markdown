---
layout: post
title:  "Resident Evil 2 Frame Breakdown"
image_sliders_load_all: true
image_sliders:
  - girl_0_g_pass
  - girl_0_occlusion_depth
  - girl_0_shadow_maps
  - girl_0_integrated_light

---
<style type="text/css">
.center-image
{
    margin: 0 auto;
    display: block;
}
</style>
{% include slider_styles.html %}

## General notes 
All results have been acquired on a pretty old setup(i7 3770+GTX 770) running on DirectX 11 with medium quality and using RenderDoc and Nsight.  
The game runs on Re Engine, a successor of MT Framework - the previous generation engine from Capcom R&D. Aside from RE2 it is used in DMC5 and RE7:Biohazard.  
Frame breakdowns are among those things I really like on the internet. I'm not a graphics programmer but I'll try my best to make this post more than just a bunch of random image dumps.

I couldn't find any material on RE Engine on the internet, so everything here is an (hopefully)educated guess. I'm covering about 90% of the frame structure here and just the general gist of the algorithms. It's really difficult to cover more as it would require more experience than I have and time to reverse engineer the shaders.

As always, check out cool links at the bottom and don't forget to reach out if you find any mistakes or have suggestions. I really like learning from the feedback.

## Table of content
* this unordered seed list will be replaced
{:toc}

## Frame Structure
### Particle/Fluid sim
Among other things ripple textures are being generated(Example from a different frame).  
![particle sim 0](/assets/re2/troop_1/particle_sim/fluid_scaled.png)
![particle sim 0](/assets/re2/troop_1/particle_sim/ripple_scaled.png)

Ripples are used for water rendering which is not present in this frame.

Some of the results are being copied into a staging buffer which suggests that results may be used by CPU. 
### Light list calculation
This pass generates visible light list by testing light frustums against the view frustum. The result is visible light list and some sort of 3d table that maps view space positions to the corresponding lights.
### White point
This pass builds a histogram of brightness based on the previous hdr image and metering table. Then determines the whitepoint based on that data.  
![metering table](/assets/re2/girl_0/whitepoint/metering_map_scaled.png)
![prevhdr](/assets/re2/girl_0/whitepoint/prev_hdr_scaled.png)

### Determine occluders
Occluders' bounding boxes are being tested against view frustum in a compute shader and an indirect argument buffer is filled.  
### Occlusion culling
Occluders are rendered into a small resolution depth buffer and then bounding boxes are being tested against this depth buffer.  

{% include slider.html selector="girl_0_occlusion_depth" %}  

Used depth buffer is 4x multisampled. Probably to compensate for low res.

![occlusion culling 0](/assets/re2/girl_0/occlusion_culling/occlusion_culling_geometry_scaled.png)

Looks like view oriented bounding boxes.  
![occlusion culling 0](/assets/re2/girl_0/occlusion_culling/occlusion_test_scaled.png)

Example of occlusion test(From a different frame). Surviving pixels(green) write flags(1 for example) to the visibility buffer with per instance slots.  
```nasm
store_raw RWDrawIndirectArguments.x, v1.x, l(1)
```
### Accumulate indirect arguments
Almost every world space geometry object is being rendered with an indirect drawcall.
Nsight profiler shows calls to NvAPI_D3D11_MultiDrawIndexedInstancedIndirect. Read \[[1]\] and \[[2]\] about its usage.
RenderDoc blocks MultiDraw extension so instead in the EventBrowser these expand into a lot of DrawIndexedInstancedIndirect where some of them are empty.  
The job of this pass is to aggregate visibility masks from the previous pass and generate an argument buffer.
### Depth prepass
Nothing fancy. Subset of the scene with major occluders.  
![depth prepass](/assets/re2/girl_0/main_depth_prepass.png)
### G-Buffer pass Geometry+Decals

{% include slider.html selector="girl_0_g_pass" %}

Output:  
* RT0 - r11g11b10 emissive
* RT1 - rgba8 albedo.rgb + metallness.a
* RT2 - r10g10b10a2 normal.rg + roughness.b + misc.a
* RT3 - r16g16b16a16 baked_ao.x + velocity.yz + sss.a

![](/assets/re2/girl_0/main_color/baked_ao_example_scaled.png)

The rendered models use pre-baked ambient occlusion from hi-res models.



### HiZ calculation
![HiZ gif](/assets/re2/girl_0/hiz_update/hiz.gif)  
Multi pass compute shader determines each level of depth hierarchy.  
### AO
SSAO or HBAO+ depending on your settings. SSAO in this case.  
AO is calculated based on HiZ from the previous pass.  
![G virus guy final form](/assets/re2/girl_0/hiz_update/ao_0_scaled.png)

### Global Specular+Diffuse
Using some non trivial algorithm light probes, cubemaps and AO are combined into global diffuse and specular maps.  

![GID](/assets/re2/girl_0/light/cubemap_0_scaled.png)
![GID](/assets/re2/girl_0/light/cubemap_1_scaled.png)

Example cubemaps from the scene.


![GID](/assets/re2/girl_0/light/GID_scaled.png)

Global illumination diffuse component.  

![GIS](/assets/re2/girl_0/light/GIS_scaled.png)

Global illumination specular component.

### Update shadowmaps
Per light shadowmaps are being updated for those lights that are affected by dynamic objects. Each shadow map is allocated on a big texture array.  

{% include slider.html selector="girl_0_shadow_maps" %}

### Local Specular+Diffuse+SSS
Per light contribution to specular and diffuse component is computed.

![GID](/assets/re2/girl_0/light/diffuse_scaled.png)

Diffuse+SSS. SSS contribution is not visible is this frame.  

![GIS](/assets/re2/girl_0/light/specular_scaled.png)

Specular component.

### Integrating the light

{% include slider.html selector="girl_0_integrated_light" %}  


### Apply transparent glass
![](/assets/re2/girl_0/light/glass.gif)

After all lightning has been applied, transparent glass is rendered.

### Compute volumetrics/haze/smoke

![](/assets/re2/girl_0/light/haze_src_scaled.png)

Basically just a bunch of sprites.

### Apply volumetrics/haze/smoke
![](/assets/re2/girl_0/light/blur_8th_scaled.png)

This pass computes the blurred image to better lit the haze.

![](/assets/re2/girl_0/check/checkerboard_haze_scaled.png)

If the original haze mask is replaced with checkerboard.

![GIS](/assets/re2/girl_0/light/haze.gif)

The result of this pass.

### TAA with previous HDR image

![GIS](/assets/re2/girl_0/light/tta.gif)

TTA is just magic.

### Motion blur

![GIS](/assets/re2/girl_0/light/neighborMax_scaled.png)

Blur guide map is computed based on velocity map.

![GIS](/assets/re2/girl_0/light/mb.gif)

### Post Processing
![GIS](/assets/re2/girl_0/light/downscaled_scaled.png)

This pass computes the downscaled image first  

![GIS](/assets/re2/girl_0/light/postprocess.gif)

And then applies bloom filter, tonemapping, distortion and chromatic abberration.

## Conclusion

The Engine relies heavily on compute+indirect draw approach. All meshes and textures are of high quality.
The game employs deferred rendering with TAA/FXAA and glass as a post process. Read [this](http://www.adriancourreges.com/blog/2015/03/10/deus-ex-human-revolution-graphics-study/) for more details on deferred rendering tricks. A lot of textures are BC7 compressed. In general, the used techniques are similar to those described at \[[4]\].

## Bonus section
### SSS contribution

![GIS](/assets/re2/girl_0/check/sss.gif)

Had to really search for the right frame to notice the SSS contribution.

### Tentacle monster breakdown
![G virus guy final form](/assets/re2/g_final.gif)

The monster is split into many parts.

### The guts

![G virus guy final form](/assets/re2/guts_0.gif)

The game has a lot of gore textures for different details.

Mip levels for textures are not just linearly filtered. The crispiness increases with each level. Also some textures already have specular highlights, not sure how this contributes to the look.  

### Geometry

![G virus guy final form](/assets/re2/tentacle_geometry.png)

Geometry has good topology.  


### HBAO

![G virus guy final form](/assets/re2/troop_high/hbao/ao_final_scaled.png)

HBAO is of much better quality than SSAO.

### Smoke

![](/assets/re2/troop_high/smoke.gif)

Smoke takes cone lights into account.

![](/assets/re2/girl_0/check/smoke_check_scaled.png)

The actual planes used to render the smoke.

{% include slider_scripts.html %}

## References
[1][Kostas Anagnostou: Experiments in GPU-based occlusion culling][1]

[1]: https://interplayoflight.wordpress.com/2018/01/15/experiments-in-gpu-based-occlusion-culling-part-2-multidrawindirect-and-mesh-lodding/

[2][Daniel Rákos: Multi-Draw-Indirect is here][2]

[2]: http://rastergrid.com/blog/2011/06/multi-draw-indirect-is-here/

[3][Adrian Courrèges' Blog][3]

[3]: http://www.adriancourreges.com/blog/

[4][Alien: Isolation][4]

[4]: https://community.amd.com/community/gaming/blog/2015/05/12/high-tech-fear--alien-isolation


