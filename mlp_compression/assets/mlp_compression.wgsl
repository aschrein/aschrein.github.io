// Based on https://github.com/boksajak/Dx12NN

@group(0) @binding(0) var<storage, read_write> g_color : array<vec4f>;
@group(0) @binding(1) var g_sampler : sampler;
@group(0) @binding(2) var g_texture : texture_2d<f32>;
// Weights and biases storage buffer
@group(0) @binding(3) var<storage, read_write> g_rw_params : array<f32>;
// Means and variance storage buffer for adam optimizer exponential moving
// averages
// @group(0) @binding(4) var<storage, read_write> g_rw_adam_params: array<f32>;
@group(0) @binding(4) var<storage, read_write> g_rw_gradients : array<atomic<i32>>;

struct Constants {
    frame_idx : u32,
};

@group(0) @binding(5) var<uniform> constants : Constants;

const NUM_LAYERS                : u32 = u32(6);
const MAX_NUM_NODES_PER_LAYER   : u32 = u32(32);
const NUM_NODES_PER_LAYER =
    array<u32, NUM_LAYERS>(
    32,
    32,
    16,
    32,
    16,
    3);
const NO_RESIDUAL = u32(0xffffffff);
const RESIDUAL_CONNECTIONS = array<u32, NUM_LAYERS>(
    NO_RESIDUAL,
    NO_RESIDUAL,
    NO_RESIDUAL,
    u32(0),
    u32(1),
    2u);
const NUM_ACTIVATIONS_PER_NETWORK : u32 = NUM_LAYERS * MAX_NUM_NODES_PER_LAYER;

fn get_total_num_nodes()->u32 {
    var total_num_nodes : u32 = 0u;
    for (var i : u32 = 0u; i < NUM_LAYERS; i = i + 1u) {
        total_num_nodes = total_num_nodes + NUM_NODES_PER_LAYER[i];
    }
    return total_num_nodes;
}

/**
Layer params memory layout

--------------------
* where N is the number of nodes in the current layer and M
    is the number of nodes in the previous layer

weights[N][M] : f32
biases[N] : f32
adam_weights_mean_variance[N][M][2] : f32
adam_biases_mean_variance[N][2] : f32

*/

struct LayerConstants {
    num_nodes : u32,
    num_prev_nodes : u32,
    num_weights : u32,
    num_biases : u32,
    num_adam_params : u32,
    num_activations : u32,
    // offsets relative to the start of the layer
    weights_offset : u32,
    biases_offset : u32,
    adam_weights_offset : u32,
    adam_biases_offset: u32,
};

fn get_layer_constants(layer_idx : u32)->LayerConstants {
    if (layer_idx == 0u) {
        return LayerConstants(
            /*num_nodes*/NUM_NODES_PER_LAYER[0],
            /*num_prev_nodes*/0u,
            /*num_weights*/0u,
            /*num_biases*/0u,
            /*num_adam_params*/0u,
            /*num_activations*/NUM_NODES_PER_LAYER[0],
            /* weights_offset*/ 0u,
            /* biases_offset*/ 0u,
            /* adam_weights_offset*/ 0u,
            /* adam_biases_offset*/ 0u,
        );
    }
    let num_prev_nodes : u32     = NUM_NODES_PER_LAYER[layer_idx - 1];
    let num_nodes : u32          = NUM_NODES_PER_LAYER[layer_idx];
    let num_weights : u32        = num_prev_nodes * num_nodes;
    let num_biases : u32         = num_nodes;
    let num_adam_params : u32    = 4 * (num_weights + num_biases);
    let num_activations : u32    = num_nodes;
    let weights_offset : u32     = 0u;
    let biases_offset : u32      = num_weights;
    let adam_weights_offset : u32 = num_weights + num_biases;
    let adam_biases_offset : u32 = num_weights + num_biases + 2 * num_weights;
    return LayerConstants(
        /*num_nodes*/ num_nodes,
        /*num_prev_nodes*/ num_prev_nodes,
        /*num_weights*/ num_weights,
        /*num_biases*/ num_biases,
        /*num_adam_params*/ num_adam_params,
        /*num_activations*/ num_activations,
        /* weights_offset*/ weights_offset,
        /* biases_offset*/ biases_offset,
        /* adam_weights_offset*/ adam_weights_offset,
        /* adam_biases_offset*/ adam_biases_offset);
}
fn get_grad_offset(layer_idx : u32) -> u32 {
    /**
        return the offset for the gradients in the storage buffer
    */
    var offset : u32 = 0u;
    for (var i : u32 = 1u; i < layer_idx; i = i + 1u) {
        let layer_constants = get_layer_constants(i);
        offset                = offset + layer_constants.num_weights + layer_constants.num_biases;
    }
    return offset;

}
fn get_layer_activations_offset(layer_idx : u32)->u32 {
    return MAX_NUM_NODES_PER_LAYER * layer_idx;

    // var offset : u32 = 0u;
    // for (var i : u32 = 0u; i < layer_idx; i = i + 1u) {
    //     let layer_constants = get_layer_constants(i);
    //     offset                = offset + layer_constants.num_nodes;
    // }
    // return offset;
}
fn get_layer_params_offset(layer_idx : u32)->u32 {
    // Get offset for a layer in the storage buffer
    // Layer 0 has no weights or biases
    // Layer 1 has weights and biases for input -> hidden layer etc.
    // layer_idx stores the mapping from previous layer to this layer
    var offset : u32 = 0u;
    for (var i : u32 = 1u; i < layer_idx; i = i + 1u) {
        // let layer_constants = get_layer_constants(i);
        let num_nodes = MAX_NUM_NODES_PER_LAYER;
        let num_weights = MAX_NUM_NODES_PER_LAYER * MAX_NUM_NODES_PER_LAYER;
        let num_biases = MAX_NUM_NODES_PER_LAYER;
        let num_adam_params = 4 * (num_weights + num_biases);
        offset = offset + num_weights + num_biases + num_adam_params;
        // offset                = offset + layer_constants.num_weights +
                //  layer_constants.num_biases + layer_constants.num_adam_params;
    }
    return offset;
}

struct CSInput {
    @builtin(global_invocation_id) global_id : vec3<u32>,
};

fn apply_gamma(color : vec3f, gamma : f32)->vec3f {
    return vec3f(pow(color.r, gamma), pow(color.g, gamma), pow(color.b, gamma));
}

const width : u32  = u32(512);
const height : u32 = u32(512u);

// low bias 32 bit random number generator
// https://github.com/skeeto/hash-prospector
fn lowbias32(_x : u32)->u32 {
    var x = _x;
    x = x ^ (x >> 16);
    x = x * 0x7feb352d;
    x = x ^ (x >> 15);
    x = x * 0x846ca68b;
    x = x ^ (x >> 16);
    return x;
}

// random number generator between 0 and 1 using 65535(0xffff) as the max value
fn random_uniform_unit_float(rnd_state : ptr<function, u32>)->f32 {
    let r : u32 = lowbias32(*rnd_state);
    *rnd_state  = r;
    return f32(r & u32(0xffff)) / f32(0xffff);
}

// clang-format off
@compute
@workgroup_size(8, 8, 1)
fn Initialize(input : CSInput) {
    // clang-format on

    let pos : vec2<u32> = input.global_id.xy;
    let idx : u32 = pos.x + pos.y * width;
    // Figure out layer index
    let layer_idx : u32 = idx / MAX_NUM_NODES_PER_LAYER;
    let node_idx : u32  = idx % MAX_NUM_NODES_PER_LAYER;

    if (layer_idx == 0u) {
        return;
    }
    if (layer_idx >= NUM_LAYERS) {
        return;
    }

    let layer_constants = get_layer_constants(layer_idx);
    
    if (node_idx >= layer_constants.num_nodes) {
        return;
    }
    
    let layer_params_offset = get_layer_params_offset(layer_idx);
    let normalization_const: f32 = 6.0 / sqrt(f32(layer_constants.num_weights));

    var rnd_state : u32 = idx;
    rnd_state = lowbias32(rnd_state);
    for (var i : u32 = 0u; i < layer_constants.num_prev_nodes; i = i + 1u) {
        let weight_offset : u32 =
                                layer_params_offset
                                + layer_constants.weights_offset
                                + node_idx * layer_constants.num_prev_nodes
                                + i;
        g_rw_params[weight_offset] = (random_uniform_unit_float(&rnd_state) * 2.0 - 1.0)
                                    * normalization_const;
    }
    let biases_offset : u32 = layer_params_offset + layer_constants.biases_offset + node_idx;
    // Initialize biases to 0
    g_rw_params[biases_offset] = (random_uniform_unit_float(&rnd_state) * 2.0 - 1.0) * normalization_const / 10.0;
    // g_rw_params[biases_offset] = 0.0;

    // Initialize adam optimizer exponential moving averages
    // for weights
    for (var i : u32 = 0u; i < layer_constants.num_prev_nodes; i = i + 1u) {
        let adam_weights_mean_offset : u32 =
                                            layer_params_offset
                                            + layer_constants.adam_weights_offset
                                            + 2 * node_idx * layer_constants.num_prev_nodes
                                            + 2 * i;
        g_rw_params[adam_weights_mean_offset + 0] = 0.0;
        g_rw_params[adam_weights_mean_offset + 1] = 0.0;
    }
    // for biases
    let adam_biases_mean_offset : u32 = layer_params_offset
                                        + layer_constants.adam_biases_offset
                                        + 2 * node_idx;
    g_rw_params[adam_biases_mean_offset + 0] = 0.0;
    g_rw_params[adam_biases_mean_offset + 1] = 0.0;

    // Initialize gradients to 0
    let grad_offset : u32 = get_grad_offset(layer_idx);
    for (var i : u32 = 0u; i < layer_constants.num_prev_nodes; i = i + 1u) {
        let grad_buf_idx : u32 = grad_offset
                                + node_idx * layer_constants.num_prev_nodes
                                + i;
        atomicStore(&g_rw_gradients[grad_buf_idx], 0);
    }
    let bias_grad_buf_idx : u32 = grad_offset + layer_constants.num_weights + node_idx;
    atomicStore(&g_rw_gradients[bias_grad_buf_idx], 0);

}

fn sigmoid(x : f32)->f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn sigmoid_derivative(x : f32)->f32 {
    return sigmoid(x) * (1.0 - sigmoid(x));
}

fn gelu(x : f32)->f32 {
    return x * sigmoid(x);
}

fn gelu_derivative(x : f32)->f32 {
    return sigmoid(x) + x * sigmoid_derivative(x);
}

fn leaky_relu(x : f32)->f32 {
    let alpha : f32 = 0.01;
    return max(alpha * x, x);
}

fn leaky_relu_derivative(x : f32)->f32 {
    let alpha : f32 = 0.01;
    if (x > 0.0) {
        return 1.0;
    } else {
        return alpha;
    }
}

fn Inference(layer_idx : u32,
             node_idx  : u32,
            activations : ptr<function, array<f32, NUM_ACTIVATIONS_PER_NETWORK>>) {

    let layer_constants     = get_layer_constants(layer_idx);
    let layer_params_offset = get_layer_params_offset(layer_idx);
    let weights_offset      = layer_params_offset + layer_constants.weights_offset;
    let biases_offset       = layer_params_offset + layer_constants.biases_offset;
    let activations_offset  = get_layer_activations_offset(layer_idx - 1);
    let activation_idx      = get_layer_activations_offset(layer_idx) + node_idx;
    var acc : f32            = 0.0;
    for (var i : u32 = 0u; i < layer_constants.num_prev_nodes; i = i + 1u) {
        let weight_idx : u32 = weights_offset
                                + node_idx * layer_constants.num_prev_nodes
                                + i;
        let weight : f32     = g_rw_params[weight_idx];
        acc                  = acc + activations[activations_offset + i] * weight;
        if (RESIDUAL_CONNECTIONS[layer_idx] != NO_RESIDUAL) {
            // check out of bounds
            acc = acc + activations[get_layer_activations_offset(RESIDUAL_CONNECTIONS[layer_idx]) + i] * weight;
        }
    }
    
    let bias_idx : u32 = biases_offset + node_idx;
    let bias : f32     = g_rw_params[bias_idx];
    
    activations[activation_idx] = leaky_relu(acc + bias);
    
        // activations[activation_idx] = acc;
}
const NUM_GRAD_QUANTIZATION_LEVELS : u32 = u32(1 << 11);
fn quantize_grad(grad : f32)->i32 {
    let num_levels : u32 = NUM_GRAD_QUANTIZATION_LEVELS;
    return i32(round(grad * f32(num_levels)));
}
fn dequantize_grad(quantized_grad : i32)->f32 {
    let num_levels : u32 = NUM_GRAD_QUANTIZATION_LEVELS;
    return f32(quantized_grad) / f32(num_levels);
}
fn Backprop(
    layer_idx : u32,
    node_idx : u32,
    activations : ptr<function, array<f32, NUM_ACTIVATIONS_PER_NETWORK>>,
    grads : ptr<function, array<f32, NUM_ACTIVATIONS_PER_NETWORK>>
) {
    let layer_constants     = get_layer_constants(layer_idx);
    let layer_params_offset = get_layer_params_offset(layer_idx);
    let weights_offset      = layer_params_offset + layer_constants.weights_offset;
    let biases_offset       = layer_params_offset + layer_constants.biases_offset;
    let grad_offset         = get_grad_offset(layer_idx);
    let activations_offset  = get_layer_activations_offset(layer_idx - 1);
    let activation_idx      = get_layer_activations_offset(layer_idx) + node_idx;
    let activation          = activations[activation_idx];
    let grad                = grads[activation_idx];
    var delta               = grad * leaky_relu_derivative(activation);

    for (var i : u32 = 0u; i < layer_constants.num_prev_nodes; i = i + 1u) {
        let weight_idx  = weights_offset
                        + node_idx * layer_constants.num_prev_nodes
                        + i;
        let weight      = g_rw_params[weight_idx];
        grads[activations_offset + i] = grads[activations_offset + i]
                        + delta * weight;
        var prev_activation : f32 = activations[activations_offset + i];
        if (RESIDUAL_CONNECTIONS[layer_idx] != NO_RESIDUAL) {
            let residual_activation_idx = get_layer_activations_offset(RESIDUAL_CONNECTIONS[layer_idx]) + i;
            prev_activation = prev_activation + activations[residual_activation_idx];
            grads[residual_activation_idx] = grads[residual_activation_idx] + delta * weight;
        }
        let grad_buf_idx = grad_offset + node_idx * layer_constants.num_prev_nodes + i;
        atomicAdd(&g_rw_gradients[grad_buf_idx], quantize_grad(delta * prev_activation));

        
        // g_rw_gradients[grad_idx] = g_rw_gradients[weight_idx] + delta * activations[activation_idx];
    }
    let bias_idx = grad_offset + layer_constants.num_weights + node_idx;
    atomicAdd(&g_rw_gradients[bias_idx], quantize_grad(delta / (f32(layer_constants.num_prev_nodes))));
}
fn get_luma(color : vec3f)->f32 {
    return 0.299 * color.r + 0.715 * color.g + 0.0722 * color.b;
}
fn getsign(x : f32)->f32 {
    if (x > 0.0) {
        return 1.0;
    } else {
        return -1.0;
    }
}
const BATCH_SIZE : u32 = (4u * 4u) * 64u;

fn InitializeActivation(
    uv : vec2<f32>,
    activations : ptr<function, array<f32, NUM_ACTIVATIONS_PER_NETWORK>>
) {
    let input_layer_idx : u32 = u32(0);
    // activations[0] = uv.x;
    // activations[1] = uv.y;
    // activations[2] = sin((uv.x + uv.y) * 3.14159);
    // activations[3] = cos((uv.x - uv.y) * 3.14159);
    // Frequency encoding of uv
    let num_frequncies : u32 = 8u; // 3 * 2 * 2 + 2 = 12 + 2 = 14
    let num_channels : u32 = 2u;
    for (var i : u32 = 0u; i < num_frequncies; i = i + 1u) {
        for (var channel_idx : u32 = 0u; channel_idx < num_channels; channel_idx = channel_idx + 1u) {
            let activation_idx : u32 =
                0
                + 2 * i * num_channels
                + 2 * channel_idx;
            let power_bias : u32 = 0;
            let sin_val : f32        = sin(uv[channel_idx] * pow(2.0, f32(power_bias + i)) * 3.14159);
            let cos_val : f32        = cos(uv[channel_idx] * pow(2.0, f32(power_bias + i)) * 3.14159);
            activations[activation_idx + 0] = sin_val / pow(1.0, f32(i));
            activations[activation_idx + 1] = cos_val / pow(1.0, f32(i));;
        }
    }

}

// clang-format off
@compute
@workgroup_size(8, 8, 1)
fn Main(input : CSInput) {
    // clang-format on
    let _pos    : vec2<u32> = input.global_id.xy % 8u;
    let _tile   : vec2<u32> = input.global_id.xy / 8u;
    // let tile_idx = _tile.x + _tile.y * 4;
    let num_tiles = 64u;
    let poffset = lowbias32(_tile.x + lowbias32(_tile.y + lowbias32(constants.frame_idx / 1))) % (num_tiles * num_tiles);
    let tile_coord = vec2<u32>(poffset % num_tiles, poffset / num_tiles);
    let pos = _pos + tile_coord * 8u;
    if (pos.x >= width || pos.y >= height) {
        return;
    }
    let uv : vec2<f32> = vec2<f32>(f32(pos.x) / f32(width), f32(pos.y) / f32(height));
    // let color   : vec4f             = textureSample(g_texture, g_sampler,
    // uv);
    let color : vec4f = textureSampleLevel(g_texture, g_sampler, uv, 0);
    // g_color[pos.x + pos.y * width]  = vec4f(uv, 0.0, 1.0);
    // array of activations
    var activations = array<f32, NUM_ACTIVATIONS_PER_NETWORK>();
    for (var i : u32 = 0u; i < NUM_ACTIVATIONS_PER_NETWORK; i = i + 1u) {
        activations[i] = 0.0;
    }
    // Initialize input layer activations
    InitializeActivation(uv, &activations);
    const start_layer_idx : u32 = u32(1); // we start from the second layer
    for (var i : u32 = start_layer_idx; i < NUM_LAYERS; i = i + 1u) {
        for (var j : u32 = 0u; j < NUM_NODES_PER_LAYER[i]; j = j + 1u) {
            Inference(i, j, &activations);
        }
    }
    let final_activation_idx : u32 = get_layer_activations_offset(NUM_LAYERS - 1);
    let final_activation_r : f32     = activations[final_activation_idx + 0];
    let final_activation_g : f32     = activations[final_activation_idx + 1];
    let final_activation_b : f32     = activations[final_activation_idx + 2];

    var grads = array<f32, NUM_ACTIVATIONS_PER_NETWORK>();
    for (var i : u32 = 0u; i < NUM_ACTIVATIONS_PER_NETWORK; i = i + 1u) {
        grads[i] = 0.0;
    }
    let target_signal = color.xyz;
    let target_luma = get_luma(target_signal);
    let src_luma    = get_luma(vec3f(final_activation_r, final_activation_g, final_activation_b));
    let luma_duff   = (src_luma - target_luma);
    let signal_diff = vec3f(final_activation_r, final_activation_g, final_activation_b) - target_signal;
    grads[final_activation_idx + 0] = 4.0 * (signal_diff[0] + 0.001 * getsign(signal_diff[0])) + 0.0 * luma_duff * 0.299;
    grads[final_activation_idx + 1] = 4.0 * (signal_diff[1] + 0.001 * getsign(signal_diff[1])) + 0.0 * luma_duff * 0.715;
    grads[final_activation_idx + 2] = 4.0 * (signal_diff[2] + 0.001 * getsign(signal_diff[2])) + 0.0 * luma_duff * 0.0722;
    // Backpropagation
    for (var i : u32 = NUM_LAYERS - 1; i >= start_layer_idx; i = i - 1u) {
        for (var j : u32 = 0u; j < NUM_NODES_PER_LAYER[i]; j = j + 1u) {
            Backprop(i, j, &activations, &grads);
        }
    }

    // g_color[pos.x + pos.y * width] = vec4f(color.xyz, 1.0);
    // g_color[pos.x + pos.y * width] = vec4f(   final_activation_r
    //                                         , final_activation_g
    //                                         , final_activation_b
    //                                         , 1.0);
                                        // let t = 4;
    // g_color[pos.x + pos.y * width] = vec4f(
        // activations[t],
        // activations[t],
        // activations[t],
    // 1.0);
}

// clang-format off
@compute @workgroup_size(8, 8, 1)
fn InferencePass(input : CSInput) {
    // clang-format on
    let pos : vec2<u32> = input.global_id.xy;
    if (pos.x >= width || pos.y >= height) {
        return;
    }
    let idx : u32 = pos.x + pos.y * width;
    let uv : vec2<f32> = vec2<f32>(f32(pos.x) / f32(width), f32(pos.y) / f32(height));
    var activations = array<f32, NUM_ACTIVATIONS_PER_NETWORK>();
    for (var i : u32 = 0u; i < NUM_ACTIVATIONS_PER_NETWORK; i = i + 1u) {
        activations[i] = 0.0;
    }
    // Initialize input layer activations
    InitializeActivation(uv, &activations);
    const start_layer_idx : u32 = u32(1); // we start from the second layer
    for (var i : u32 = start_layer_idx; i < NUM_LAYERS; i = i + 1u) {
        for (var j : u32 = 0u; j < NUM_NODES_PER_LAYER[i]; j = j + 1u) {
            Inference(i, j, &activations);
        }
    }
    let final_activation_idx : u32 = get_layer_activations_offset(NUM_LAYERS - 1);
    let final_activation_r : f32     = activations[final_activation_idx + 0];
    let final_activation_g : f32     = activations[final_activation_idx + 1];
    let final_activation_b : f32     = activations[final_activation_idx + 2];

    g_color[pos.x + pos.y * width] = vec4f(
          final_activation_r
        , final_activation_g
        , final_activation_b
        , 1.0);
}

fn GetLearningRate()->f32 {
    let t = saturate(f32(constants.frame_idx) / f32(1 << 10));
    let up_slope = 64.0 * t;
    let down_slope = cos(t * 3.14159) * 0.5 + 0.5;
    // let BATCH_SIZE : u32 = u32(512*512);
    return max(min(up_slope,down_slope), 0.001) * 0.8 / sqrt(f32(BATCH_SIZE));
}

// clang-format off
@compute
@workgroup_size(8, 8, 1)
fn Backward(input : CSInput) {
    // clang-format on

    let pos : vec2<u32> = input.global_id.xy;
    let idx : u32 = pos.x + pos.y * width;
    // Figure out layer index
    let layer_idx   : u32 = idx / MAX_NUM_NODES_PER_LAYER;
    let node_idx    : u32 = idx % MAX_NUM_NODES_PER_LAYER;
    if (layer_idx == 0u) {
        return;
    }
    if (layer_idx >= NUM_LAYERS) {
        return;
    }

    let layer_constants = get_layer_constants(layer_idx);
    
    if (node_idx >= layer_constants.num_nodes) {
        return;
    }

    let layer_params_offset     : u32 = get_layer_params_offset(layer_idx);
    let grad_offset             : u32 = get_grad_offset(layer_idx);
    let adam_weights_offset     : u32 = layer_params_offset + layer_constants.adam_weights_offset;
    let adam_biases_offset      : u32 = layer_params_offset + layer_constants.adam_biases_offset;

    let lr : f32 = GetLearningRate();

    // Update weights
    for (var i : u32 = 0u; i < layer_constants.num_prev_nodes; i = i + 1u) {
        let weight_idx : u32 = layer_params_offset
                                + layer_constants.weights_offset
                                + node_idx * layer_constants.num_prev_nodes
                                + i;
        let grad_buf_idx : u32 = grad_offset
                                + node_idx * layer_constants.num_prev_nodes
                                + i;
        let grad : f32 = dequantize_grad(atomicExchange(&g_rw_gradients[grad_buf_idx], 0));
        let adam_weight_idx : u32 = adam_weights_offset
                                    + 2 * node_idx * layer_constants.num_prev_nodes
                                    + 2 * i;
        let adam_mean       : f32 = g_rw_params[adam_weight_idx + 0];
        let adam_variance   : f32 = g_rw_params[adam_weight_idx + 1];
        let rnd = f32(lowbias32(i + lowbias32(node_idx + lowbias32(constants.frame_idx / 1))) & 0xffff) / f32(0xffff);
        var adam_mean_new   : f32 = mix(adam_mean, grad, rnd * 0.5); // TODO: use momentum
        if (abs(adam_mean_new) < 1.0e-5) {
            adam_mean_new = grad;
        }
        g_rw_params[weight_idx] = g_rw_params[weight_idx] - grad * lr;
        g_rw_params[adam_weights_offset + 2 * weight_idx + 0] = adam_mean_new;

    }
    // Update biases
    let bias_idx               : u32 = layer_params_offset + layer_constants.biases_offset + node_idx;
    let bias_grad_buf_idx      : u32 = grad_offset + layer_constants.num_weights + node_idx;
    let bias_grad              : f32 = dequantize_grad(atomicExchange(&g_rw_gradients[bias_grad_buf_idx], 0));
    let adam_bias_idx          : u32 = adam_biases_offset + 2 * node_idx;
    let adam_bias_mean         : f32 = g_rw_params[adam_bias_idx + 0];
    let adam_bias_variance     : f32 = g_rw_params[adam_bias_idx + 1];
    var adam_bias_mean_new     : f32 = mix(adam_bias_mean, bias_grad, 0.02); // TODO: use momentum
    if (abs(adam_bias_mean_new) < 1.0e-5) {
        adam_bias_mean_new = bias_grad;
    }
    g_rw_params[bias_idx] = g_rw_params[bias_idx] - adam_bias_mean_new * lr * 0.008;
    g_rw_params[adam_bias_idx + 0] = adam_bias_mean_new;
}