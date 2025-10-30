use anyhow::bail;
use const_format::formatcp;
use objc2::{AllocAnyThread, Encode, Message};
use objc2_metal::MTLCopyAllDevices;
use regex_automata::dfa::Automaton;
use tracing::{info, info_span, instrument};

use block2::RcBlock;

use crate::shift_and_dist::{ShiftAndDist, ShiftAndDistAutomaton, BLOCK_SIZE};
use regex_syntax::parse;

use std::fmt::Debug;
use std::fmt::Write;
use std::future::Future;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{
    NSArray, NSObjectNSKeyValueCoding, NSObjectProtocol, NSRange, NSString, NSURL,
};
use objc2_metal::{
    MTLAllocation, MTLArgumentBuffersTier, MTLArgumentDescriptor, MTLArgumentEncoder,
    MTLBindingAccess, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLDispatchType,
    MTLFunction, MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandType, MTLIndirectComputeCommand, MTLIndirectRenderCommand, MTLLibrary,
    MTLLibraryOptimizationLevel, MTLPipelineOption, MTLResource, MTLResourceOptions,
    MTLResourceUsage, MTLSize,
};

use anyhow::{anyhow, Context, Result};

// The size of the block to process on one thread we set it to one page size (execute pagesize in terminal)
const MAX_INPUT_SIZE: usize = usize::pow(2, 21);

// const SHADER_PATH: &str = concat!(env!("OUT_DIR"), "/shift_and_dist_optimized.metallib");
// const SHADER_PATH: &str = concat!(env!("OUT_DIR"), "/shift_and_dist.metallib");
const SIMD_WIDTH: u32 = 32;
const THREADGROUP_SIZE: u32 = 32;

type Id<T> = Retained<ProtocolObject<T>>;

#[allow(dead_code)]
#[derive(Debug)]
pub struct ShiftAndDistGPU {
    pattern: String,
    automaton: Box<ShiftAndDistAutomaton>,
    device: Id<dyn MTLDevice>,
    command_queue: Id<dyn MTLCommandQueue>,
    compute_pipeline_state: Id<dyn MTLComputePipelineState>,
    automaton_buffer: Id<dyn MTLBuffer>,
}

const SHADER_PATH: &str = concat!(env!("OUT_DIR"), "/shift_and_dist.metallib");

#[allow(dead_code)]
impl ShiftAndDistGPU {
    #[instrument]
    pub fn new(pattern: &str, count_matches_per_line: bool) -> Self {
        // #[cfg(feature = "argument_buffer")]
        // println!("Using argument buffer");
        // #[cfg(feature = "dispatch_threads")]
        // println!("Using dispatch threads");
        // #[cfg(feature = "indirect_command_buffer")]
        // println!("Using indirect command buffer");

        let hir = parse(pattern).unwrap();

        let automaton = Box::new(Self::compile_hir(&hir).unwrap());

        let (device, command_queue) = Self::setup_gpu().context("Setting up GPU").unwrap();

        let kernel_function = Self::read_shader_from_file(&device, SHADER_PATH, "shift_and_dist")
            .context("Failed to read shader from file")
            .unwrap();

        let pipeline_desc = MTLComputePipelineDescriptor::new();
        pipeline_desc.setLabel(Some(&NSString::from_str("shift_and_dist")));
        pipeline_desc.setComputeFunction(Some(&kernel_function));

        let compute_pipeline_state = unsafe {
            device.newComputePipelineStateWithDescriptor_options_reflection_error(
                &pipeline_desc,
                MTLPipelineOption::None,
                None,
            )
        }
        .context("failed to create metal compute pipeline state")
        .unwrap();

        // let automaton_pointer = NonNull::new(std::ptr::from_ref(&automaton) as *mut c_void)
        //     .context("Failed to create automation pointer")?;
        // automaton.as_ref()

        let automaton_buffer = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(automaton.as_ref() as *const _ as *mut std::ffi::c_void).unwrap(),
                std::mem::size_of::<ShiftAndDistAutomaton>(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .context("Failed to create automaton pointer")
        .unwrap();

        println!("setup done");

        Self {
            pattern: pattern.to_string(),
            automaton,
            device,
            command_queue,
            compute_pipeline_state,
            automaton_buffer,
        }
    }

    #[instrument]
    fn setup_gpu() -> Result<(Id<dyn MTLDevice>, Id<dyn MTLCommandQueue>)> {
        // let device = MTLCreateSystemDefaultDevice().context("Failed to create Metal device")?;
        let device = MTLCopyAllDevices();
        let device = device.firstObject().context("Failed to get Metal device")?;

        let command_queue = device
            .newCommandQueue()
            .context("Failed to create Metal command queue")?;

        Ok((device, command_queue))
    }

    #[instrument]
    fn read_shader_from_file(
        device: &Id<dyn MTLDevice>,
        file_path: &str,
        main_function: &str,
    ) -> Result<Id<dyn MTLFunction>> {
        let library_url = unsafe { NSURL::fileURLWithPath(&NSString::from_str(file_path)) };

        let library = unsafe { device.newLibraryWithURL_error(&library_url) }
            .map_err(|e| anyhow!(e.localizedDescription().to_string()))
            .context("Failed to create Metal library")?;

        let kernel_function = library
            .newFunctionWithName(&NSString::from_str(main_function))
            .context("Failed to create Metal kernel function")?;

        Ok(kernel_function)
    }

    /// Generates the core distance calculation code for the GPU shader
    ///
    /// This function creates the part of the shader that performs the
    /// shift-and algorithm's distance calculations.
    pub fn generate_distance_gpu_code(automaton: &ShiftAndDistAutomaton) -> Result<String> {
        let mut code = String::with_capacity(256);
        code.push_str("((");

        for d in 0..=automaton.max_dist {
            if automaton.masks_dist[d] != 0 {
                writeln!(
                    code,
                    "((state & 0x{:x}) << {d}) | ",
                    automaton.masks_dist[d]
                )?;
            }
        }

        // Add the final mask application
        writeln!(code, "{}) & masks_char[c]);", automaton.mask_initial)?;

        Ok(code)
    }

    /// Generates a C array declaration for character masks used in the shader
    ///
    /// Creates a constant array containing the automaton's character masks
    /// for the relevant character range.
    pub fn generate_char_masks_array(
        automaton: &ShiftAndDistAutomaton,
        state_type: &str,
        first_char_index: usize,
        last_char_index: usize,
    ) -> Result<String> {
        if last_char_index <= first_char_index {
            bail!("Invalid character range: last_char_index must be greater than first_char_index");
        }

        if last_char_index > automaton.masks_char.len() {
            bail!("last_char_index exceeds automaton character mask array length");
        }

        let mut code = String::with_capacity(last_char_index * 8);
        write!(
            code,
            "constant {state_type} masks_char[{last_char_index}] = {{"
        )
        .context("Failed to format array declaration")?;

        for c in 0..last_char_index {
            if automaton.masks_char[c] == 0 {
                code.push_str("0, ");
            } else {
                write!(code, "0x{:x}, ", automaton.masks_char[c])?;
            }
        }

        code.push_str("};\n");
        Ok(code)
    }

    /// Finds the smallest appropriate type for representing the automaton state
    fn determine_state_type(automaton: &ShiftAndDistAutomaton) -> &'static str {
        // Find the first most left bit set to 1 in all of the automaton's char masks
        let state_size = 64
            - automaton
                .masks_char
                .iter()
                .fold(0, |acc, &x| acc | x)
                .leading_zeros();

        match state_size {
            0..=8 => "uint8_t",
            9..=16 => "uint16_t",
            17..=32 => "uint32_t",
            _ => "uint64_t",
        }
    }

    /// Determines the character range that needs to be processed
    fn find_char_range(automaton: &ShiftAndDistAutomaton) -> (usize, usize) {
        // Find the first character with a non-zero mask
        let first_char_index = automaton
            .masks_char
            .iter()
            .enumerate()
            .find(|(_, &x)| x != 0)
            .map_or(0, |(i, _)| i);

        // Find the last character with a non-zero mask
        let last_char_index = automaton
            .masks_char
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &x)| x != 0)
            .map_or(0, |(i, _)| i);

        (first_char_index, last_char_index)
    }

    #[instrument]
    fn setup_compute_command_encoder(
        &self,
        input_ptr: *const u8,
        input_len: usize,
        num_blocks: usize,
    ) -> Result<(
        Id<dyn MTLCommandBuffer>,
        Id<dyn MTLComputeCommandEncoder>,
        Id<dyn MTLBuffer>,
    )> {
        let input_buffer = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    NonNull::new(input_ptr as *mut std::ffi::c_void)
                        .context("Failed to create input pointer")?,
                    size_of::<u8>() * input_len,
                    MTLResourceOptions::StorageModeShared
                        | MTLResourceOptions::CPUCacheModeWriteCombined
                        | MTLResourceOptions::HazardTrackingModeUntracked,
                    None,
                )
        }
        .context("Failed to create input buffer")?;

        let output_buffer = self
            .device
            .newBufferWithLength_options(
                std::mem::size_of::<u8>() * num_blocks,
                MTLResourceOptions::StorageModeShared,
            )
            .context("Failed to create output buffer")?;

        let command_buffer = self
            .command_queue
            .commandBuffer()
            .context("Failed to create Metal command buffer")?;

        let compute_command_encoder = command_buffer
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .context("Failed to get compute command encoder")?;

        compute_command_encoder.setComputePipelineState(&self.compute_pipeline_state);
        unsafe {
            compute_command_encoder.setBuffer_offset_atIndex(Some(&self.automaton_buffer), 0, 0);
            compute_command_encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 1);
            compute_command_encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 2);
        }

        Ok((command_buffer, compute_command_encoder, output_buffer))
    }

    #[inline(always)]
    fn dispatch_command_encoder(
        &self,
        compute_command_encoder: &Id<dyn MTLComputeCommandEncoder>,
        input_len: usize,
    ) {
        let max = self.compute_pipeline_state.maxTotalThreadsPerThreadgroup();
        assert_eq!(max, (SIMD_WIDTH * THREADGROUP_SIZE).try_into().unwrap());

        let num_threadgroup_threads = MTLSize {
            width: SIMD_WIDTH as usize,
            height: THREADGROUP_SIZE as usize,
            depth: 1,
        };
        let threads_per_threadgroup = (num_threadgroup_threads.width
            * num_threadgroup_threads.height
            * num_threadgroup_threads.depth) as u32;

        let num_threadgroups = MTLSize {
            width: (input_len as f32 / threads_per_threadgroup as f32).ceil() as usize,
            height: 1,
            depth: 1,
        };
        compute_command_encoder
            .dispatchThreadgroups_threadsPerThreadgroup(num_threadgroups, num_threadgroup_threads);
    }
}

impl ShiftAndDist for ShiftAndDistGPU {
    fn count_matches(&mut self, text: &str) -> usize {
        let mut input = text.bytes().collect::<Vec<u8>>();
        input.push(0x17);
        let input = input.into_boxed_slice();

        // We do a small trick here, setting the output buffer to 4 * u8 to get the 32bits
        let (command_buffer, compute_command_encoder, output_buffer) = self
            .setup_compute_command_encoder(input.as_ptr(), input.len(), 4)
            .unwrap();

        // let num_threadgroups = MTLSize {
        //     width: 1,
        //     height: 1,
        //     depth: 1,
        // };
        // let num_groups = MTLSize {
        //     width: 1,
        //     height: 1,
        //     depth: 1,
        // };

        self.dispatch_command_encoder(&compute_command_encoder, 1);

        compute_command_encoder.endEncoding();
        command_buffer.commit();

        unsafe { command_buffer.waitUntilCompleted() };

        let output_ptr = output_buffer.contents().as_ptr() as *const u32;

        let output = unsafe { std::slice::from_raw_parts(output_ptr, 1) }[0] as usize;

        output
    }

    #[instrument]
    fn count_match_lines(&self, lines: &[[u8; BLOCK_SIZE]]) -> Result<usize> {
        let input_len = lines.len();
        assert!(input_len <= MAX_INPUT_SIZE, "Too many lines");

        let (command_buffer, compute_command_encoder, output_buffer) = self
            .setup_compute_command_encoder(
                lines.as_ptr().cast::<u8>(),
                input_len * BLOCK_SIZE,
                input_len,
            )
            .unwrap();

        // let execution_span = info_span!("gpu_execution").entered();
        self.dispatch_command_encoder(&compute_command_encoder, input_len);

        compute_command_encoder.endEncoding();
        command_buffer.commit();

        unsafe { command_buffer.waitUntilCompleted() };

        // execution_span.exit();

        // let kernel_start_time = unsafe { command_buffer.kernelStartTime() };
        // let kernel_end_time = unsafe { command_buffer.kernelEndTime() };
        // let gpu_start_time = unsafe { command_buffer.GPUStartTime() };
        // let gpu_end_time = unsafe { command_buffer.GPUEndTime() };
        // println!(
        //     "Time {} ms = GPU {} ms + Kernel {} ms + Diff {} ms",
        //     (gpu_end_time - kernel_start_time) * 1_000f64,
        //     (gpu_end_time - gpu_start_time) * 1_000f64,
        //     (kernel_end_time - kernel_start_time) * 1_000f64,
        //     (gpu_start_time - kernel_end_time) * 1_000f64
        // );

        // let result_processing_span = info_span!("result_processing").entered();
        let output_ptr = output_buffer.contents().as_ptr() as *const u8;
        let output: Vec<u8> = unsafe { std::slice::from_raw_parts(output_ptr, input_len).to_vec() };
        let count = output.into_iter().filter(|c| *c > 0).count();
        // info!("matched_lines = {}", count);
        // result_processing_span.exit();

        Ok(count)
    }

    #[instrument]
    fn match_lines_future(
        &self,
        lines: &[[u8; BLOCK_SIZE]],
    ) -> Result<impl Future<Output = Vec<bool>>> {
        let input_len = lines.len();
        assert!(input_len <= MAX_INPUT_SIZE, "Too many lines");

        let (command_buffer, compute_command_encoder, output_buffer) = self
            .setup_compute_command_encoder(
                lines.as_ptr().cast::<u8>(),
                input_len * BLOCK_SIZE,
                input_len,
            )
            .unwrap();

        let execution_span = info_span!("gpu_execution").entered();

        self.dispatch_command_encoder(&compute_command_encoder, input_len);

        compute_command_encoder.endEncoding();

        let shared_data = Arc::new(Mutex::new(ShiftAndFutureData {
            result: None,
            waker: None,
            output_buffer: Some(output_buffer),
        }));

        let shared_data_clone = shared_data.clone();

        let handler = RcBlock::new(
            move |command_buffer_ptr: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let command_buffer = unsafe { command_buffer_ptr.as_ref() };

                let kernel_start_time = unsafe { command_buffer.kernelStartTime() };
                let kernel_end_time = unsafe { command_buffer.kernelEndTime() };
                let gpu_start_time = unsafe { command_buffer.GPUStartTime() };
                let gpu_end_time = unsafe { command_buffer.GPUEndTime() };
                println!(
                    "Time {} ms = GPU {} ms + Kernel {} ms + Diff {} ms",
                    (gpu_end_time - kernel_start_time) * 1_000f64,
                    (gpu_end_time - gpu_start_time) * 1_000f64,
                    (kernel_end_time - kernel_start_time) * 1_000f64,
                    (gpu_start_time - kernel_end_time) * 1_000f64
                );

                let mut data = shared_data_clone.lock().unwrap();
                if let Some(output_buffer) = data.output_buffer.as_ref() {
                    let output_ptr = output_buffer.contents().as_ptr() as *const u8;
                    let output: &[u8] =
                        unsafe { std::slice::from_raw_parts(output_ptr, input_len) };
                    data.result = Some(output.iter().map(|&c| c > 0).collect());
                }

                if let Some(waker) = data.waker.take() {
                    waker.wake();
                }
            },
        );

        unsafe {
            command_buffer.addCompletedHandler(RcBlock::into_raw(handler));
        }

        command_buffer.commit();

        execution_span.exit();

        Ok(ShiftAndFuture { data: shared_data })
    }
}

struct ShiftAndFutureData {
    result: Option<Vec<bool>>,
    waker: Option<Waker>,
    output_buffer: Option<Id<dyn MTLBuffer>>,
}

// Trust me I know what I'm doing :)
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for ShiftAndFutureData {}
unsafe impl Sync for ShiftAndFutureData {}

struct ShiftAndFuture {
    data: Arc<Mutex<ShiftAndFutureData>>,
}

impl Future for ShiftAndFuture {
    type Output = Vec<bool>;
    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let mut data = self.data.lock().unwrap();

        data.result.take().map_or_else(
            || {
                data.waker = Some(cx.waker().clone());
                Poll::Pending
            },
            Poll::Ready,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[ignore]
    #[test]
    fn test_count_simple() {
        // env_logger::builder()
        //     .filter_level(log::LevelFilter::Debug)
        //     .format_timestamp_nanos()
        //     .init();

        let pattern = "ab{0,2}c";
        let text = "abbcabc";
        let expected = 2;
        let mut sa = ShiftAndDistGPU::new(pattern, true);
        println!("Setup Done");
        let matches = sa.count_matches(text);
        info!("matches = {}", matches);
        info!("matches2 = {}", sa.count_matches("ab"));
        assert_eq!(matches, expected, "Pattern: {pattern}, Text: {text}");
    }

    fn test_count_matches(pattern: &str, text: &str, expected: usize) {
        let mut sa = ShiftAndDistGPU::new(pattern, true);
        // let (mask_final, masks_char, masks_dist, max_dist) =
        //     ShiftAndDistGPU::compile_tokens(parse(pattern).unwrap()).unwrap();
        // println!("mask_final = {:04b}", mask_final);
        // println!("masks_char['a'] = {:04b}", masks_char['a' as usize]);
        // println!("masks_char['b'] = {:04b}", masks_char['b' as usize]);
        // println!("masks_char['c'] = {:04b}", masks_char['c' as usize]);
        // println!("masks_dist[0] = {:04b}", masks_dist[0]);
        // println!("masks_dist[1] = {:04b}", masks_dist[1]);
        // println!("masks_dist[2] = {:04b}", masks_dist[2]);
        // println!("max_dist = {}", max_dist);

        let matches = sa.count_matches(text);
        assert_eq!(matches, expected, "Pattern: {pattern}, Text: {text}");
    }

    #[test]
    fn test_count_matches_range() {
        test_count_matches("ab{1,2}c", "ac", 0);
        test_count_matches("ab{1,2}c", "abc", 1);
        test_count_matches("ab{0,2}c", "abbc", 1);
        test_count_matches("ab{0,2}c", "abbbc", 0);
        test_count_matches("ab{0,2}c", "abbbbc", 0);
        test_count_matches("ab{0,2}c", "abbcabc", 2);
        test_count_matches("ab{0,2}c", "acac", 2);
        test_count_matches("ab{1,2}c", "abcac", 1);
    }

    #[test]
    fn test_count_matches_plus() {
        test_count_matches("ab+c", "ac", 0);
        test_count_matches("ab+c", "abc", 1);
        test_count_matches("ab+c", "abbc", 1);
        test_count_matches("ab+c", "abbbbbbc", 1);
        test_count_matches("ab+c", "abbcabc", 2);
        test_count_matches("ab+c", "acac", 0);
        test_count_matches("ab+c", "abcac", 1);
    }

    #[test]
    fn test_count_matches_star() {
        test_count_matches("ab*c", "ac", 1);
        test_count_matches("ab*c", "abc", 1);
        test_count_matches("ab*c", "abbc", 1);
        test_count_matches("ab*c", "abbbbbbc", 1);
        test_count_matches("ab*c", "abbcabc", 2);
        test_count_matches("ab*c", "acac", 2);
        test_count_matches("ab*c", "abcac", 2);
    }

    #[test]
    fn test_count_matches_nested() {
        test_count_matches("a((b*c){1,2}d){0,3}e", "ae", 1);
        test_count_matches("a((b*c){1,2}d){0,3}e", "acde", 1);
        test_count_matches("a((b*c){1,2}d){0,3}e", "abbbcdeae", 2);
        test_count_matches("a((b*c){1,2}d){0,3}e", "aeae", 2);
    }

    #[test]
    fn test_count_matches_repeating() {
        test_count_matches("ab{0,2}a", "abaa", 2);
        // abaa aaa aaba abaa
        test_count_matches("ab{0,2}ab{0,2}a", "abaaabaa", 4);
    }

    #[test]
    fn test_count_alternation() {
        test_count_matches("a[b-g]z", "acz", 1);
        test_count_matches("a[b-g]z", "apz", 0);
        test_count_matches("a(bc|de)f", "abcf", 1);
        test_count_matches("a(bc|de)f", "abef", 0);
    }

    #[test]
    fn test_count_matches_multi_end() {
        test_count_matches("a(b|c)", "ac", 1);
        test_count_matches("a(b|c)", "ab", 1);
        test_count_matches("a(b|c)", "ad", 0);
    }

    #[test]
    fn test_count_matches_multi_start() {
        test_count_matches("(b|c)d", "cd", 1);
        test_count_matches("(ab|cd)e", "abe", 1);
        test_count_matches("(ab|cd)e", "cde", 1);
        test_count_matches("(ab|cd)ef", "cbe", 0);
    }
}
