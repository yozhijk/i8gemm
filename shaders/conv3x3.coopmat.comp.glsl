#version 460 core

// --- Required Extensions ---
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_8bit_storage: require

const int TILE_SIZE_X = 8;
const int TILE_SIZE_Y = 4;
const int TILE_SIZE_X_WITH_RING = TILE_SIZE_X + 2;
const int TILE_SIZE_Y_WITH_RING = TILE_SIZE_Y + 2;
const int TILE_SIZE = TILE_SIZE_X * TILE_SIZE_Y;
const int TILE_SIZE_WITH_RING = TILE_SIZE_X_WITH_RING * TILE_SIZE_Y_WITH_RING;

const int WORKGROUP_SIZE_X = 32;
const int WORKGROUP_SIZE_Y = 4;
const int WORKGROUP_SIZE = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y;

const int SIZE_OF_IVEC4 = 16;
const int IC_SLICE = 32;
const int OC_SLICE = 16;
const int IC_SLICE_IN_IVEC4 = IC_SLICE / SIZE_OF_IVEC4;
const int OC_SLICE_IN_IVEC4 = OC_SLICE / SIZE_OF_IVEC4;

// 3D tensor TILE_SIZE_Y_WITH_RING x TILE_SIZE_X_WITH_RING x IC_SLIZE 
const int INPUT_TILE_SIZE_IN_BYTES = TILE_SIZE_WITH_RING * IC_SLICE;
const int INPUT_TILE_SIZE_IN_IVEC4 = INPUT_TILE_SIZE_IN_BYTES / SIZE_OF_IVEC4;

shared ivec4 shared_tile_ivec4[INPUT_TILE_SIZE_IN_IVEC4];

layout(local_size_x = WORKGROUP_SIZE_X, local_size_y = WORKGROUP_SIZE_Y, local_size_z = 1) in;

// Input / output bindings
layout(set = 0, binding = 0) readonly buffer Input {
    ivec4 t_input[]; 
};

layout(set = 0, binding = 1) readonly buffer Weight {
    int8_t t_weight[];
};

layout(set = 0, binding = 2) buffer Output {
    int t_output[];
};

layout(push_constant) uniform PushConsts {
    uint num_ic;
    uint num_oc;
    uint height;
    uint width;
    uint pad[4];
} p;

void load_tile_to_shared(uint tile_start_x, uint tile_start_y, uint ic_slice_start)
{
    // Input is in HWC, each channel slice is consecutive chunk of SIZE_OF_IVEC4 * IC_SLICE / 16 bytes
    // ic_slice is 0..(num_input_channels / IC_SLICE)-1
    uint linear_idx = gl_LocalInvocationID.y * WORKGROUP_SIZE_X + gl_LocalInvocationID.x;

    // we need to load INPUT_TILE_SIZE_IN_IVEC4 ivec4 elements,
    // in case of IC_SLICE == 16 we load the whole slice in single ivec4 load.
    // but in general each slice needs IC_SLICE / SIZE_OF_IVEC4 loads.
    // it makes sense to assume IC_SLICE >= SIZE_OF_IVEC4.
    while (linear_idx < INPUT_TILE_SIZE_IN_IVEC4)
    {
        // calculate local row col and channel for current linear_idx
        uint row = linear_idx / (TILE_SIZE_X_WITH_RING * IC_SLICE_IN_IVEC4);
        uint col = (linear_idx % (TILE_SIZE_X_WITH_RING * IC_SLICE_IN_IVEC4)) / IC_SLICE_IN_IVEC4;
        uint ch  = (linear_idx % (TILE_SIZE_X_WITH_RING * IC_SLICE_IN_IVEC4)) % IC_SLICE_IN_IVEC4;

        // calculate global row col and channel
        int g_row = int(tile_start_y + row - 1);
        int g_col = int(tile_start_x + col - 1);

        // channels are indexed in ivec4 units
        uint g_ic_slice = ic_slice_start / SIZE_OF_IVEC4 + ch;
        uint g_row_stride = (p.num_ic / SIZE_OF_IVEC4) * p.width;
        uint g_col_stride = (p.num_ic / SIZE_OF_IVEC4);
  
        if (g_row >=0 && g_row < p.height && g_col >=0 && g_col < p.width)
        {
            shared_tile_ivec4[linear_idx] = t_input[g_row_stride * g_row + g_col_stride * g_col + g_ic_slice];
        }
        else
        {
            shared_tile_ivec4[linear_idx] = ivec4(0);
        }

        linear_idx += WORKGROUP_SIZE;
    }
}


// Get offset in ivec4 elements of a matrix corresponding to 
// p kernel position for subgroup_id row of pixels. 
uint get_shmem_offset_3x3(uint p, uint subgroup_id) {
    uint row = p / 3;
    uint col = p % 3;
    uint row_stride = TILE_SIZE_X_WITH_RING * IC_SLICE / SIZE_OF_IVEC4;

    return (row + subgroup_id) * row_stride + col * IC_SLICE / SIZE_OF_IVEC4;
}


// Tiled implicit Conv3x3 kernel. 
// Runs TILE_SIZE_Y subgroups, each subgroup is responsible for TILE_SIZE_X input pixels,
// each represented by IC_SLICE input channels. I.e. each subgroup runs 9
// (TILE_SIZE_X x IC_SLICE) @ (IC_SLIICE x OC_SLICE) MMA primtives (for each spatial kernel position).
// Whole group handles TILE_SIZE_X x TILE_SIZE_Y input pixels tile. 
// The group iterates over input slices. Output slices can be either iterated as well or 
// mapped to  Z workgroup coordinate and run in parallel.
void main() {
    // These matrices essentially occupy size * gl_NumSubgroups
    coopmat<int8_t, gl_ScopeSubgroup, TILE_SIZE_X, IC_SLICE, gl_MatrixUseA> mat_a;
    coopmat<int8_t, gl_ScopeSubgroup, IC_SLICE, OC_SLICE, gl_MatrixUseB> mat_b;
    coopmat<int, gl_ScopeSubgroup, TILE_SIZE_X, OC_SLICE, gl_MatrixUseAccumulator> mat_c;

    // The kernel handles TILE_SIZE_X x TILE_SIZE_Y tile by chunking input and output channels
    // w.r.t to MMA primitive size. 
    uint tile_start_y = gl_WorkGroupID.y * TILE_SIZE_Y;
    uint tile_start_x = gl_WorkGroupID.x * TILE_SIZE_X;

    // Output slice loop
    // TODO: this can be mapped to WG z coordinated in order 
    // to reuse input tile cached in shared memory for multiple output slices.
    // Try this. 
    for (uint os = 0; os < p.num_oc / OC_SLICE; ++os)
    {
        // Clear C matrix.
        for (int i = 0; i < mat_c.length(); ++i)
        {
	        mat_c[i] = 0;
        }

        // Input tile loop
        for (uint is = 0; is < p.num_ic / IC_SLICE; ++is)
        {
            // Load input channels slice to shared memory.
            load_tile_to_shared(tile_start_x, tile_start_y, is * IC_SLICE);
            barrier();

            // Iterate over kernel weight for each spatial position (3x3)
            for (uint k = 0; k < 9; ++k)
            {
                // Load matrix B for position p of the kernel from memory (weight matrix)
                // weights are laid out in NUM_OUT_SLICESxNUM_INPUT_SLICESxKxKxIC_SLICExOC_SLICE
                // (64, 64, 3, 3) -> (4, 4, 3, 3, 16, 16).

                // Stride between kernel positions
                uint P_STRIDE = IC_SLICE * OC_SLICE;
                // Stride between input slices
                uint IC_STRIDE = P_STRIDE * 9;
                // Stride between output slices
                uint OC_STRIDE  = IC_STRIDE * (p.num_ic / IC_SLICE);

                uint offset = os * OC_STRIDE + is * IC_STRIDE + k * P_STRIDE;
                coopMatLoad(mat_b, t_weight, offset, OC_SLICE, gl_CooperativeMatrixLayoutRowMajor);

                // Load matrix A for position p:
                // each workgroup row (subgroup) loads different row of pixels from shared memory,
                // accounting for 3x3 kernel spatial offset.
                coopMatLoad(mat_a, shared_tile_ivec4, get_shmem_offset_3x3(k, gl_LocalInvocationID.y), IC_SLICE / SIZE_OF_IVEC4, gl_CooperativeMatrixLayoutRowMajor);

                // Perform MMA and accumulate to mat_c.
                mat_c = coopMatMulAdd(mat_a, mat_b, mat_c);
            }

            // Make sure not to overwrite input tile, while some warps are still doing MMAs.
            barrier();
        }

        // Store output channel slice.
        uint OUT_COL_STRIDE = p.num_oc;
        uint OUT_ROW_STRIDE = p.width * OUT_COL_STRIDE; 

        // Each subgroup stores different row of output data.
        uint orow = (tile_start_y + gl_LocalInvocationID.y);
        uint ocol = tile_start_x;
        uint output_offset = orow * OUT_ROW_STRIDE + ocol * OUT_COL_STRIDE + os * OC_SLICE;

        coopMatStore(mat_c, t_output, output_offset, OUT_COL_STRIDE, gl_CooperativeMatrixLayoutRowMajor); 
    }
}
