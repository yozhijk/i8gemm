#version 460 core
// --- Required Extensions ---
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shared_memory_block: require

// NOTE: Depending on your glslang version, you might need this for explicit float8 types.
// If your compiler is very new, it might support float8_t directly. 
// Otherwise, we often treat storage as uint8_t and cast during load, 
// OR simply define the matrix inputs as uint8_t and the hardware config handles the interpretation.
// #extension GL_EXT_shader_explicit_arithmetic_types_float8 : enable 

// --- Constants (Specialization Constants are better for prod) ---
// Cooperative Matrix Size: 16x16 is the most standard cross-vendor tile size.
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

// 3D tensor TILE_SIZE_Y_WITH_RING x TILE_SIZE_X_WITH_RING x IC_SLIZE elements.
const int INPUT_TILE_SIZE_IN_BYTES = TILE_SIZE_WITH_RING * IC_SLICE;
const int INPUT_TILE_SIZE_IN_IVEC4 = INPUT_TILE_SIZE_IN_BYTES / SIZE_OF_IVEC4;

shared Memory
{
    layout(offset=0) int8_t shared_tile_i8[INPUT_TILE_SIZE_IN_BYTES];
    layout(offset=0, align=16) ivec4 shared_tile_ivec4[INPUT_TILE_SIZE_IN_IVEC4];
} shmem;


layout(local_size_x = WORKGROUP_SIZE_X, local_size_y = WORKGROUP_SIZE_Y, local_size_z = 1) in;

// --- Buffers ---
layout(set = 0, binding = 0) readonly buffer Input {
    ivec4 t_input[]; 
};

layout(set = 0, binding = 1) readonly buffer Weight {
    int8_t t_weight[];
};

layout(set = 0, binding = 2) buffer Output {
    int t_output[];
};

// Push Constants for dynamic dimensions
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
        uint g_ic_slice = ic_slice_start + ch;
        uint g_row_stride = (p.num_ic / IC_SLICE) * p.width;
        uint g_col_stride = (p.num_ic / IC_SLICE);

        if (g_row >=0 && g_row < p.height && g_col >=0 && g_col < p.width)
        {
            shmem.shared_tile_ivec4[linear_idx] = t_input[g_row * g_row_stride + g_col * g_col_stride + g_ic_slice];
        }
        else
        {
            shmem.shared_tile_ivec4[linear_idx] = ivec4(0);
        }

        linear_idx += WORKGROUP_SIZE;
    }
}

uint get_shmem_offset_3x3(uint p, uint subgroup_id) {
    uint row = p / 3;
    uint col = p % 3;
    uint row_stride = TILE_SIZE_X_WITH_RING * IC_SLICE;

    return (row + subgroup_id) * row_stride + col * IC_SLICE;
}


void main() {
    // These matrices essentially occupy size * gl_NumSubgroups
    coopmat<int8_t, gl_ScopeSubgroup, TILE_SIZE_X, IC_SLICE, gl_MatrixUseA> mat_a;
    coopmat<int8_t, gl_ScopeSubgroup, IC_SLICE, OC_SLICE, gl_MatrixUseB> mat_b;
    coopmat<int, gl_ScopeSubgroup, TILE_SIZE_X, OC_SLICE, gl_MatrixUseAccumulator> mat_c;

    uint tile_start_y = gl_WorkGroupID.y * TILE_SIZE_Y;
    uint tile_start_x = gl_WorkGroupID.x * TILE_SIZE_X;


    // Input slice loop
    for (uint os = 0; os < p.num_oc / OC_SLICE; ++os)
    {
        mat_c = coopmat<int, gl_ScopeSubgroup, TILE_SIZE_X, OC_SLICE, gl_MatrixUseAccumulator>(0);

        for (uint is = 0; is < p.num_ic / IC_SLICE; ++is)
        {
            // 1. Load input slice to shared memory
            load_tile_to_shared(tile_start_x, tile_start_y, is * IC_SLICE_IN_IVEC4);
            barrier();

            for (uint k = 0; k < 9; ++k)
            {
                // 2. Load matrix B for position p of the kernel from memory (weight matrix)
                // weights are laid out in NUM_OUT_SLICESxNUM_INPUT_SLICESxIN_SLICExKxKxIC_SLICExOC_SLICE
                // (64, 64, 3, 3) -> (4, 4, 3, 3, 16, 16)
                uint P_STRIDE = IC_SLICE * OC_SLICE;
                uint IC_STRIDE = P_STRIDE * 9;
                uint OC_STRIDE  = IC_STRIDE * (p.num_ic / IC_SLICE);
                uint offset = os * OC_STRIDE + is * IC_STRIDE + k * P_STRIDE;
                coopMatLoad(mat_b, t_weight, offset, OC_SLICE, gl_CooperativeMatrixLayoutRowMajor);

                // 4. Load matrix A for position p for row r 
                coopMatLoad(mat_a, shmem.shared_tile_i8, get_shmem_offset_3x3(k, gl_LocalInvocationID.y), IC_SLICE, gl_CooperativeMatrixLayoutRowMajor);
                barrier();

                // 5. Perform MMA and accumulate to mat_c[r]
                mat_c = coopMatMulAdd(mat_a, mat_b, mat_c);
                barrier();
            }
        }

        // Store output slice
        uint OUT_COL_STRIDE = p.num_oc;
        uint OUT_ROW_STRIDE = p.width * OUT_COL_STRIDE; 
        uint orow = (tile_start_y + gl_LocalInvocationID.y);
        uint ocol = tile_start_x;
        uint output_offset = orow * OUT_ROW_STRIDE + ocol * OUT_COL_STRIDE + os * OC_SLICE;
        //mat_c = coopmat<int, gl_ScopeSubgroup, TILE_SIZE_X, OC_SLICE, gl_MatrixUseAccumulator>(orow);
        coopMatStore(mat_c, t_output, output_offset, OUT_COL_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
        //t_output[output_offset] = 42;
    }
}
