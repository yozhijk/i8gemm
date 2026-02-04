#version 460 core

// --- Required Extensions ---
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_8bit_storage: require

// Intel Lunar Lake MMA primitive: 8x32x16 (M=8, K=32, N=16)
// 32x4 tile - best config, each subgroup computes 4 adjacent 8-wide outputs

const int TILE_SIZE_X = 32;
const int TILE_SIZE_Y = 4;
const int TILE_SIZE_X_WITH_RING = TILE_SIZE_X + 2;
const int TILE_SIZE_Y_WITH_RING = TILE_SIZE_Y + 2;
const int TILE_SIZE_WITH_RING = TILE_SIZE_X_WITH_RING * TILE_SIZE_Y_WITH_RING;

const int WORKGROUP_SIZE_X = 32;
const int WORKGROUP_SIZE_Y = 4;
const int WORKGROUP_SIZE = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y;

const int SIZE_OF_IVEC4 = 16;
const int IC_SLICE = 32;
const int OC_SLICE = 16;
const int IC_SLICE_IN_IVEC4 = IC_SLICE / SIZE_OF_IVEC4;

const int MMA_M = 8; 

const int INPUT_TILE_SIZE_IN_BYTES = TILE_SIZE_WITH_RING * IC_SLICE;
const int INPUT_TILE_SIZE_IN_IVEC4 = INPUT_TILE_SIZE_IN_BYTES / SIZE_OF_IVEC4;

shared ivec4 shared_tile_ivec4[INPUT_TILE_SIZE_IN_IVEC4];

layout(local_size_x = WORKGROUP_SIZE_X, local_size_y = WORKGROUP_SIZE_Y, local_size_z = 1) in;

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
    uint linear_idx = gl_LocalInvocationID.y * WORKGROUP_SIZE_X + gl_LocalInvocationID.x;
    uint g_row_stride = (p.num_ic / SIZE_OF_IVEC4) * p.width;
    uint g_col_stride = (p.num_ic / SIZE_OF_IVEC4);

    for (; linear_idx < INPUT_TILE_SIZE_IN_IVEC4; linear_idx += WORKGROUP_SIZE)
    {
        uint row = linear_idx / (TILE_SIZE_X_WITH_RING * IC_SLICE_IN_IVEC4);
        uint col = (linear_idx % (TILE_SIZE_X_WITH_RING * IC_SLICE_IN_IVEC4)) / IC_SLICE_IN_IVEC4;
        uint ch  = (linear_idx % (TILE_SIZE_X_WITH_RING * IC_SLICE_IN_IVEC4)) % IC_SLICE_IN_IVEC4;

        int g_row = int(tile_start_y + row - 1);
        int g_col = int(tile_start_x + col - 1);
        uint g_ic_slice = ic_slice_start / SIZE_OF_IVEC4 + ch;
  
        if (g_row >=0 && g_row < p.height && g_col >=0 && g_col < p.width)
        {
            shared_tile_ivec4[linear_idx] = t_input[g_row_stride * g_row + g_col_stride * g_col + g_ic_slice];
        }
        else
        {
            shared_tile_ivec4[linear_idx] = ivec4(0);
        }
    }
}


// Get shared memory offset for 3x3 kernel position k
// For 16-wide tile, x_offset selects which 8-wide half (0 or 8)
uint get_shmem_offset_3x3(uint k, uint subgroup_id, uint x_offset) {
    uint row = k / 3;
    uint col = k % 3;
    uint row_stride = TILE_SIZE_X_WITH_RING * IC_SLICE / SIZE_OF_IVEC4;
    return (row + subgroup_id) * row_stride + (col + x_offset) * IC_SLICE / SIZE_OF_IVEC4;
}


void main() {
    // Four sets of matrices for 4 parts of the 32-wide tile
    // |---------|----------|---------|----------|  
    // |    A0   |    A1    |    A2   |    A3    | 
    // |---------|----------|---------|----------| 
    //                                             

    coopmat<int8_t, gl_ScopeSubgroup, MMA_M, IC_SLICE, gl_MatrixUseA> mat_a0;
    coopmat<int8_t, gl_ScopeSubgroup, MMA_M, IC_SLICE, gl_MatrixUseA> mat_a1;
    coopmat<int8_t, gl_ScopeSubgroup, MMA_M, IC_SLICE, gl_MatrixUseA> mat_a2;
    coopmat<int8_t, gl_ScopeSubgroup, MMA_M, IC_SLICE, gl_MatrixUseA> mat_a3;
    coopmat<int8_t, gl_ScopeSubgroup, IC_SLICE, OC_SLICE, gl_MatrixUseB> mat_b;
    coopmat<int, gl_ScopeSubgroup, MMA_M, OC_SLICE, gl_MatrixUseAccumulator> mat_c0;
    coopmat<int, gl_ScopeSubgroup, MMA_M, OC_SLICE, gl_MatrixUseAccumulator> mat_c1;
    coopmat<int, gl_ScopeSubgroup, MMA_M, OC_SLICE, gl_MatrixUseAccumulator> mat_c2;
    coopmat<int, gl_ScopeSubgroup, MMA_M, OC_SLICE, gl_MatrixUseAccumulator> mat_c3;

    uint tile_start_y = gl_WorkGroupID.y * TILE_SIZE_Y;
    uint tile_start_x = gl_WorkGroupID.x * TILE_SIZE_X;

    uint subgroup_id = gl_LocalInvocationID.y;
    uint tile_z = gl_WorkGroupID.z;

    // Precompute strides
    uint P_STRIDE = IC_SLICE * OC_SLICE;
    uint IC_STRIDE = P_STRIDE * 9;
    uint OC_STRIDE = IC_STRIDE * (p.num_ic / IC_SLICE);

    uint num_oc_slices = p.num_oc / OC_SLICE;
    uint num_ic_slices = p.num_ic / IC_SLICE;

    // Precompute output strides
    uint OUT_COL_STRIDE = p.num_oc;
    uint OUT_ROW_STRIDE = p.width * OUT_COL_STRIDE; 
    uint orow = tile_start_y + subgroup_id;
    uint output_base = orow * OUT_ROW_STRIDE + tile_start_x * OUT_COL_STRIDE;

    // Output slice loop
    //for (uint os = 0; os < num_oc_slices; ++os)
    uint os = tile_z;
    {
        // Clear C matrices
        for (int i = 0; i < mat_c0.length(); ++i)
        {
            mat_c0[i] = 0;
            mat_c1[i] = 0;
            mat_c2[i] = 0;
            mat_c3[i] = 0;
        }

        uint weight_os_base = os * OC_STRIDE;

        // Input tile loop
        for (uint is = 0; is < num_ic_slices; ++is)
        {
            load_tile_to_shared(tile_start_x, tile_start_y, is * IC_SLICE);
            barrier();

            uint weight_is_offset = weight_os_base + is * IC_STRIDE;

            // Unrolled 3x3 kernel - load all 4 parts
            for (uint k = 0; k < 9; ++k)
            {
                coopMatLoad(mat_a0, shared_tile_ivec4, get_shmem_offset_3x3(k, subgroup_id, 0), IC_SLICE / SIZE_OF_IVEC4, gl_CooperativeMatrixLayoutRowMajor);
                coopMatLoad(mat_a1, shared_tile_ivec4, get_shmem_offset_3x3(k, subgroup_id, 8), IC_SLICE / SIZE_OF_IVEC4, gl_CooperativeMatrixLayoutRowMajor);
                coopMatLoad(mat_a2, shared_tile_ivec4, get_shmem_offset_3x3(k, subgroup_id, 16), IC_SLICE / SIZE_OF_IVEC4, gl_CooperativeMatrixLayoutRowMajor);
                coopMatLoad(mat_a3, shared_tile_ivec4, get_shmem_offset_3x3(k, subgroup_id, 24), IC_SLICE / SIZE_OF_IVEC4, gl_CooperativeMatrixLayoutRowMajor);
                coopMatLoad(mat_b, t_weight, weight_is_offset + k * P_STRIDE, OC_SLICE, gl_CooperativeMatrixLayoutRowMajor);
                mat_c0 = coopMatMulAdd(mat_a0, mat_b, mat_c0);
                mat_c1 = coopMatMulAdd(mat_a1, mat_b, mat_c1);
                mat_c2 = coopMatMulAdd(mat_a2, mat_b, mat_c2);
                mat_c3 = coopMatMulAdd(mat_a3, mat_b, mat_c3);
            }

            barrier();
        }

        coopMatStore(mat_c0, t_output, output_base + os * OC_SLICE, OUT_COL_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
        coopMatStore(mat_c1, t_output, output_base + 8 * OUT_COL_STRIDE + os * OC_SLICE, OUT_COL_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
        coopMatStore(mat_c2, t_output, output_base + 16 * OUT_COL_STRIDE + os * OC_SLICE, OUT_COL_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
        coopMatStore(mat_c3, t_output, output_base + 24 * OUT_COL_STRIDE + os * OC_SLICE, OUT_COL_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
    }
}
