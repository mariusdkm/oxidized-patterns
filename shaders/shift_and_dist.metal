#include <metal_stdlib>
using namespace metal;

struct Automaton {
    uint64_t max_dist;
    uint64_t mask_start;
    uint64_t mask_initial;
    uint64_t mask_final;
    uint64_t masks_char[256];
    uint64_t masks_dist[256];
};

#define SIMD_SIZE 32
#define THREADGROUP_SIZE 1024
#define BLOCK_SIZE 16384

//#define COUNT

kernel void shift_and_dist(device const Automaton& automaton [[buffer(0)]],
                           device uint8_t* output [[buffer(1)]],
                           device const uint8_t* input [[buffer(2)]],
                           uint3 group_id [[threadgroup_position_in_grid]],
                           uint3 local_id [[thread_position_in_threadgroup]]) {
    const uint block_no = group_id.x * THREADGROUP_SIZE +  local_id.y * SIMD_SIZE + local_id.x;
    const uint start = block_no * BLOCK_SIZE;
    const uint end = start + BLOCK_SIZE;

    uint64_t states = 0;
    uint matches = 0;

    uint64_t max_dist = automaton.max_dist;
    uint64_t mask_final = automaton.mask_final;
    uint64_t mask_initial = automaton.mask_initial;
    
    for (uint i = start; i < end; i++) {
        uint64_t next = mask_initial;
        for (uint d = 0; d <= max_dist; d++) {
            next |= (states & automaton.masks_dist[d]) << d;
        }
        
        char c = input[i];
        states = next & automaton.masks_char[c];
#ifdef COUNT
        matches += (states & mask_final) != 0;
#else
        if (states & mask_final) {output[block_no] = 1; return; }
#endif
        if (c == 0x17) { break; } // End of input
    }
    
    output[block_no] = matches;
}


//#include <metal_stdlib>
//using namespace metal;
//
//struct Automaton {
//    uint64_t max_dist;
//    uint64_t mask_initial;
//    uint64_t mask_final;
//    uint64_t masks_char[128];
//    uint64_t masks_dist[128];
//};
//
//#define SIMD_SIZE 32
//#define THREADGROUP_SIZE 1024
//#define BLOCK_SIZE 16384
//
//kernel void shift_and_dist(device const Automaton& automaton [[buffer(0)]],
//                           device uint32_t* output [[buffer(1)]],
//                           device const uint8_t* input [[buffer(2)]],
//                           uint3 group_id [[threadgroup_position_in_grid]],
//                           uint3 local_id [[thread_position_in_threadgroup]]) {
//    const uint block_no = group_id.x * THREADGROUP_SIZE +  local_id.y * SIMD_SIZE + local_id.x;
//    const uint start = block_no * BLOCK_SIZE;
//    const uint end = start + BLOCK_SIZE;
//
//    uint64_t states = 0;
//
//    uint64_t max_dist = automaton.max_dist;
//    uint64_t mask_final = automaton.mask_final;
//    uint64_t mask_initial = automaton.mask_initial;
//    
//    for (uint i = start; i < end; i++) {
//        uint64_t next = mask_initial;
//        for (uint d = 0; d <= max_dist; d++) {
//            next |= (states & automaton.masks_dist[d]) << d;
//        } 
//        
//        char c = input[i];
//        states = next & automaton.masks_char[c];
//        if (states & mask_final) {output[block_no] = 1; return; }
//    }
//}

//kernel void shift_and_dist_simd(device const Params& params [[buffer(0)]],
//                           device const Automaton& automaton [[buffer(1)]],
//                           device uint32_t* output [[buffer(2)]],
//                           device const uint8_t* input [[buffer(3)]],
//                           uint3 group_id [[threadgroup_position_in_grid]],
//                           uint3 local_id [[thread_position_in_threadgroup]]) {
//    const uint block_no = group_id.x * THREADGROUP_SIZE/*params.threads_per_threadgroup */+ local_id.x;
//    
//    if (block_no >= 147782) {
//        return;
//    }
//    
//   uint64_t states = 0;
//    
//    
//    const uint start = block_no * 4 * 14592/*params.block_size*/;
//    const uint end = start + 14592/*params.block_size*/;
//    
//    threadgroup uint32_t tg_masks_char[128];
//    if (local_id.x < 8) {  // 128/32 = 4 entries per thread
//        for (uint i = 0; i < 16; ++i) {
//            const uint idx = local_id.x * 16 + i;
//            uint32_t m = automaton.masks_char[idx];
//            tg_masks_char[idx] = m;
//        }
//    }
//    threadgroup_barrier(mem_flags::mem_threadgroup);
//    
//    packed_uint4 states = {0, 0, 0, 0};
//    packed_uint4 matches = {0, 0, 0, 0};
//    
//    for (uint i = start; i < end; i++) {
////        packed_uint4 total = {0, 0, 0, 0};
////        #pragma unroll
////        for (uint j = 0; j < 4; j+= 16) {
////        const uint j = 0;
//            const packed_uint4 c = { tg_masks_char[input[i + 0]], tg_masks_char[input[i + 16]], tg_masks_char[input[i + 32]], tg_masks_char[input[i + 48]]};
//            
//            states = (((states << 0) & 0x294a) |
//                      ((states << 1) & 0x1fffe) |
//                      ((states << 2) & 0x5294) |
//                      ((states << 3) & 0x8420) |
//                      ((states << 6) & 0x10000) |
//                      ((states << 11) & 0x10000) |
//                      ((states << 16) & 0x10000) |
//                      0x1) & c;
////            total += popcount(states & 0x10000);
////            
////        }
//        matches += popcount(states & 0x10000);
//    }
//    
//    output[block_no] = matches[0] + matches[1] + matches[2] + matches[3];
//}


//#include <metal_stdlib>
//using namespace metal;
//
//constant uint8_t masks_char[256] = {0, ..., 0, 0x1, 0x2, 0x2, 0, ... 0, };
//kernel void shift_and_dist(device const uint8_t* input [[buffer(0)]], device uint8_t* output [[buffer(1)]], uint2 position [[thread_position_in_grid]]) {
//    const uint block_no = position.y * THREADGROUP_SIZE + position.x;
//    uint8_t state = 0;
//    input = input + (block_no * BLOCK_SIZE);
//    for (uint i = 0; i < BLOCK_SIZE; i++) {
//        uint c = input[i];
//        state = (((state & 0x1) << 1) | 1) & masks_char[c];
//        if (state & 0x2) {output[block_no] = 1; return; }if (c == 0x17) { break; }
//    }
//}
//
//
//#include <metal_stdlib>
//using namespace metal;
//#define THREADGROUP_SIZE 32
//#define BLOCK_SIZE 16384
//constant uint8_t masks_char[256] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x1, 0x2, 0x2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
//kernel void shift_and_dist(device const uint8_t* input [[buffer(0)]], device uint8_t* output [[buffer(1)]], uint3 group_id [[threadgroup_position_in_grid]], uint3 local_id [[thread_position_in_threadgroup]]) {
//    const uint block_no = group_id.x * THREADGROUP_SIZE * THREADGROUP_SIZE + local_id.y * THREADGROUP_SIZE + local_id.x;
//    input = input + (block_no * BLOCK_SIZE);
//    uint8_t state = 0;
//    for (uint i = 0; i < BLOCK_SIZE; i++) {
//        uint c = input[i];
//        state = (((state & 0x1) << 1) | 1) & masks_char[c];
//        if (state & 0x2) { output[block_no] = 1; return; }
//        if (c == 0x17) { break; }
//    }
//}
