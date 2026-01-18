#version 460 core

// --- Required Extensions ---
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

// NOTE: Depending on your glslang version, you might need this for explicit float8 types.
// If your compiler is very new, it might support float8_t directly. 
// Otherwise, we often treat storage as uint8_t and cast during load, 
// OR simply define the matrix inputs as uint8_t and the hardware config handles the interpretation.
// #extension GL_EXT_shader_explicit_arithmetic_types_float8 : enable 

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// --- Constants (Specialization Constants are better for prod) ---
// Cooperative Matrix Size: 16x16 is the most standard cross-vendor tile size.
const int M_TILE = 16;
const int N_TILE = 16;
const int K_TILE = 16; 

// --- Buffers ---
// A: MxK matrix (Row Major), stored as 8-bit values
layout(set = 0, binding = 0) readonly buffer BufA {
    uint8_t dataA[]; 
};

// B: KxN matrix (Column Major is usually faster for B, but let's assume Row Major for simplicity)
layout(set = 0, binding = 1) readonly buffer BufB {
    uint8_t dataB[];
};

// C: MxN matrix (Row Major), stored as FP16 or FP32
layout(set = 0, binding = 2) buffer BufC {
    float16_t dataC[];
};

// Push Constants for dynamic dimensions
layout(push_constant) uniform PushConsts {
    uint M;
    uint N;
    uint K;
} p;

void main() {
    // 1. Define Matrix Types
    //    Scope: Subgroup (Standard for Vulkan)
    //    Storage: We use uint8_t for A/B to represent the raw 8 bits.
    //             The hardware "Cooperative Matrix Properties" must match this signature.
    //             (A=Uint8, B=Uint8, C=Float16, Result=Float16)
    //
    //    NOTE: If your driver exposes specific float8 e4m3 types in GLSL, use them here.
    //          Many drivers currently map (uint8, uint8) -> float16 ops if the 
    //          CooperativeMatrixProperties say "A_Type = FLOAT8_E4M3".
    coopmat<uint8_t,   gl_ScopeSubgroup, M_TILE, K_TILE, gl_MatrixUseA> matA;
    coopmat<uint8_t,   gl_ScopeSubgroup, K_TILE, N_TILE, gl_MatrixUseB> matB;
    coopmat<float16_t, gl_ScopeSubgroup, M_TILE, N_TILE, gl_MatrixUseAccumulator> matC;

    // 2. Initialize Accumulator (C) to 0
    matC = coopmat<float16_t, gl_ScopeSubgroup, M_TILE, N_TILE, gl_MatrixUseAccumulator>(0.0);

    // 3. Tile Coordinates
    //    (Assume grid is dispatched such that gl_WorkGroupID covers the matrix dimensions)
    uint globalRow = gl_WorkGroupID.y * M_TILE;
    uint globalCol = gl_WorkGroupID.x * N_TILE;

    // 4. Loop over K dimension
    for (uint k = 0; k < p.K; k += K_TILE) {
        // --- Load A ---
        // Need to verify bounds if K is not multiple of 16
        // Address: Row * Stride + Col
        // For A (RowMajor): dataA[globalRow...][k...]
        // Note: coopMatLoad requires a base index and a stride (elements per row)
        // Stride for A = K (total columns)
        uint idxA = globalRow * p.K + k;
        
        // Load params: (matrix_obj, buffer_array, start_index, stride, layout)
        coopMatLoad(matA, dataA, idxA, p.K, gl_CooperativeMatrixLayoutRowMajor);

        // --- Load B ---
        // For B (assume RowMajor KxN): dataB[k...][globalCol...]
        // Stride for B = N
        uint idxB = k * p.N + globalCol;
        
        // Using RowMajor for B is often slower/trickier for hardware, but valid.
        coopMatLoad(matB, dataB, idxB, p.N, gl_CooperativeMatrixLayoutRowMajor);

        // --- Multiply & Accumulate ---
        // Result = A * B + C
        matC = coopMatMulAdd(matA, matB, matC);
    }

    // 5. Store Result
    uint idxC = globalRow * p.N + globalCol;
    coopMatStore(matC, dataC, idxC, p.N, gl_CooperativeMatrixLayoutRowMajor);
}
