#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
in vec2 f_uv;
uniform sampler2D iChannel0;
uniform int frame_id;
uniform vec2 _resolution;
layout(location = 0) out vec4 SV_TARGET0;
vec2 iResolution;
float iTime;
float laplace(vec2 coord) {
  float stepx = 1.0;
  float stepy = 1.0;
  float l00 =
      texture(iChannel0, (coord + vec2(-stepx, -stepy)) / iResolution.xy).x;
  float l01 =
      texture(iChannel0, (coord + vec2(stepx, -stepy)) / iResolution.xy).x;
  float l10 =
      texture(iChannel0, (coord + vec2(stepx, stepy)) / iResolution.xy).x;
  float l11 =
      texture(iChannel0, (coord + vec2(-stepx, stepy)) / iResolution.xy).x;
  return (l00 + l01 + l10 + l11) * 0.25 -
         texture(iChannel0, coord / iResolution.xy).x;
}

float rand(vec2 co) {
  return fract(sin(dot(co.xy, vec2(192.9898, 78.233))) * 43758.5453);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  float offset = 0.0;
  float rand_step = 5.0;
  vec2 cell_center = 2.0 * rand_step * floor(fragCoord / rand_step / 2.0) +
                     vec2(rand_step, rand_step);
  if (iTime < 10.0) {
    if (length(fragCoord.xy - iResolution.xy * 0.5) < 8.0) {
      offset = 1.0 * sin(2.0 * iTime);
    }
    if (length(fragCoord.xy - iResolution.xy * vec2(0.2, 0.5)) < 4.0) {
      offset = 0.5 * sin(4.0 * iTime);
    }
    if (length(fragCoord.xy - iResolution.xy * vec2(0.8, 0.5)) < 4.0) {
      offset = 0.45 * sin(1.0 * iTime);
    }
  } else if (rand(cell_center / iResolution.xy +
                  rand(vec2(iTime * 777.0, -iTime))) > 0.996) {
    float l = length(fragCoord - cell_center);
    if (l < rand_step)
      offset = -0.7 * pow((rand_step - l) / rand_step, 2.0);
  }
  vec2 cur_val = texture(iChannel0, fragCoord / iResolution.xy).xy;
  float lap = laplace(fragCoord);
  cur_val.y += lap;
  float clamp_val = 100.0;
  fragColor =
      vec4(clamp(cur_val.x + cur_val.y * 0.2 + offset, -clamp_val, clamp_val),
           clamp(cur_val.y * 0.99, -clamp_val, clamp_val), 0.0, 1.0);
}
void main() {
  iTime = float(frame_id);

  vec4 fragColor;
  mainImage(fragColor, gl_FragCoord.xy);
  SV_TARGET0 = fragColor;
}
