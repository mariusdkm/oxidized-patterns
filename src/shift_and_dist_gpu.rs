use anyhow::{anyhow, Context, Result};
use block2::RcBlock;
use const_format::formatcp;
use objc2::{AllocAnyThread, Encode, Message};
use objc2_foundation::NSString;
use objc2_metal::MTLIndirectCommandBuffer;
use objc2_metal::{MTLCopyAllDevices, MTLCreateSystemDefaultDevice};
use regex_syntax::parse;
use std::fmt::Debug;
use std::fmt::Write;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};
use tracing::{info, info_span, instrument};

use crate::metal_strategy::{
    ArgumentBufferStrategy, BasicStrategy, CombinedStrategy, IndirectCommandBufferStrategy,
    MetalStrategy, ShaderVariables,
};
use crate::shift_and_dist::{ShiftAndDist, ShiftAndDistAutomaton, BLOCK_SIZE};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSObjectNSKeyValueCoding, NSObjectProtocol, NSRange, NSURL};
use objc2_metal::{
    MTLAllocation, MTLArgumentBuffersTier, MTLArgumentDescriptor, MTLArgumentEncoder,
    MTLBindingAccess, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLDispatchType, MTLFunction, MTLLibrary,
    MTLLibraryOptimizationLevel, MTLPipelineOption, MTLResource, MTLResourceOptions,
    MTLResourceUsage, MTLSize,
};

// The size of the block to process on one thread we set it to one page size (execute pagesize in terminal)
const MAX_INPUT_SIZE: usize = usize::pow(2, 21);
const THREADGROUP_SIZE: u32 = 32;

type Id<T> = Retained<ProtocolObject<T>>;

/// Different execution strategies available for Metal
#[derive(Debug, Clone, Copy)]
pub enum ExecutionStrategy {
    Basic,
    ArgumentBuffer,
    IndirectCommandBuffer,
    Combined,
}

/// Generic implementation of ShiftAndDistGPU with strategy as a type parameter
#[derive(Debug)]
pub struct ShiftAndDistGPU<S: MetalStrategy> {
    device: Id<dyn MTLDevice>,
    command_queue: Id<dyn MTLCommandQueue>,
    compute_pipeline_state: Id<dyn MTLComputePipelineState>,
    icb: Option<(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    _strategy: PhantomData<S>,
}

// Type aliases for common configurations
pub type ShiftAndDistGPUBasic = ShiftAndDistGPU<BasicStrategy>;
pub type ShiftAndDistGPUArgBuffer = ShiftAndDistGPU<ArgumentBufferStrategy>;
pub type ShiftAndDistGPUIcb = ShiftAndDistGPU<IndirectCommandBufferStrategy>;
pub type ShiftAndDistGPUCombined = ShiftAndDistGPU<CombinedStrategy>;

impl ShiftAndDistGPU<BasicStrategy> {
    /// Create a new instance with the basic strategy
    #[instrument]
    pub fn new(pattern: &str, count_matches_per_line: bool) -> Self {
        Self::new_with_strategy(pattern, count_matches_per_line)
    }
}

impl<S: MetalStrategy> ShiftAndDistGPU<S> {
    /// Create a new instance with a specific strategy
    #[instrument]
    pub fn new_with_strategy(pattern: &str, count_matches_per_line: bool) -> Self {
        // Log the strategy configuration
        let vars = S::shader_variables();
        if vars.use_argument_buffer {
            println!("Using argument buffer");
        }
        if vars.use_indirect_command_buffer {
            println!("Using indirect command buffer");
        }
        println!("Using dispatch threads");

        // Parse the pattern and create the automaton
        let hir = parse(pattern).unwrap();
        let automaton = Box::new(Self::compile_hir(&hir).unwrap());

        // Set up GPU resources
        let (device, command_queue) = Self::setup_gpu().context("Setting up GPU").unwrap();

        // Verify argument buffer support
        assert_eq!(
            device.argumentBuffersSupport(),
            MTLArgumentBuffersTier::Tier2
        );

        // Compile shader from automaton
        let library =
            Self::compile_shader_from_automaton(&device, &automaton, count_matches_per_line, vars)
                .unwrap();

        // Create pipeline state
        let kernel_function = library
            .newFunctionWithName(&NSString::from_str("shift_and_dist"))
            .context("Failed to create Metal kernel function")
            .unwrap();

        let pipeline_desc = MTLComputePipelineDescriptor::new();
        pipeline_desc.setLabel(Some(&NSString::from_str("shift_and_dist")));
        pipeline_desc.setComputeFunction(Some(&kernel_function));

        // Support indirect command buffers if needed
        if vars.use_indirect_command_buffer {
            pipeline_desc.setSupportIndirectCommandBuffers(true);
        }

        let compute_pipeline_state = unsafe {
            device.newComputePipelineStateWithDescriptor_options_reflection_error(
                &pipeline_desc,
                MTLPipelineOption::None,
                None,
            )
        }
        .context("failed to create metal compute pipeline state")
        .unwrap();

        let icb = S::create_resources(&device, &compute_pipeline_state)
            .context("Failed to create strategy resources")
            .unwrap();

        println!("setup done");

        Self {
            device,
            command_queue,
            compute_pipeline_state,
            icb,
            _strategy: PhantomData,
        }
    }

    // /// Create a new instance with a specific execution strategy
    // pub fn with_execution_strategy(pattern: &str, count_matches_per_line: bool, strategy: ExecutionStrategy) -> Result<Box<dyn Self>> {
    //     match strategy {
    //         ExecutionStrategy::Basic => {
    //             Ok(Box::new(ShiftAndDistGPU::<BasicStrategy>::new_with_strategy(
    //                 pattern, count_matches_per_line
    //             )))
    //         }
    //         ExecutionStrategy::ArgumentBuffer => {
    //             Ok(Box::new(ShiftAndDistGPU::<ArgumentBufferStrategy>::new_with_strategy(
    //                 pattern, count_matches_per_line
    //             )))
    //         }
    //         ExecutionStrategy::IndirectCommandBuffer => {
    //             Ok(Box::new(ShiftAndDistGPU::<IndirectCommandBufferStrategy>::new_with_strategy(
    //                 pattern, count_matches_per_line
    //             )))
    //         }
    //         ExecutionStrategy::Combined => {
    //             Ok(Box::new(ShiftAndDistGPU::<CombinedStrategy>::new_with_strategy(
    //                 pattern, count_matches_per_line
    //             )))
    //         }
    //     }
    // }

    #[instrument]
    fn setup_gpu() -> Result<(Id<dyn MTLDevice>, Id<dyn MTLCommandQueue>)> {
        let device = MTLCopyAllDevices();
        let device = device.firstObject().context("Failed to get Metal device")?;

        let command_queue = device
            .newCommandQueue()
            .context("Failed to create Metal command queue")?;

        Ok((device, command_queue))
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
            return Err(anyhow!(
                "Invalid character range: last_char_index must be greater than first_char_index"
            ));
        }

        if last_char_index > automaton.masks_char.len() {
            return Err(anyhow!(
                "last_char_index exceeds automaton character mask array length"
            ));
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

    /// Creates the shader header with includes and type definitions
    fn generate_shader_header(
        automaton: &ShiftAndDistAutomaton,
        state_type: &str,
        first_char_index: usize,
        last_char_index: usize,
        output_type: &str,
        vars: ShaderVariables,
    ) -> Result<String> {
        let mut header = String::with_capacity(512);

        // Standard includes and definitions
        header.push_str("#include <metal_stdlib>\n");
        header.push_str("using namespace metal;\n");
        writeln!(header, "#define THREADGROUP_SIZE {THREADGROUP_SIZE}")?;
        writeln!(header, "#define BLOCK_SIZE {BLOCK_SIZE}")?;

        // Character masks array
        let char_masks = Self::generate_char_masks_array(
            automaton,
            state_type,
            first_char_index,
            last_char_index + 1,
        )?;
        header.push_str(&char_masks);

        // Add argument buffer if needed
        if vars.use_argument_buffer {
            header.push_str("struct ArgumentBuffer {\n");
            header.push_str("    device const uint8_t* input [[id(0)]];\n");
            writeln!(header, "    device {output_type}* output [[id(1)]];")?;
            header.push_str("};\n");
        }

        Ok(header)
    }

    /// Creates the shader kernel function signature
    fn generate_kernel_signature(output_type: &str, vars: ShaderVariables) -> Result<String> {
        let input_output_type = if vars.use_argument_buffer {
            "device ArgumentBuffer & args [[buffer(0)]]".to_string()
        } else {
            format!(
                "device const uint8_t* input [[buffer(0)]], device {output_type}* output [[buffer(1)]]"
            )
        };

        // Always use dispatch threads in our refactored version
        let position_params = "uint2 position [[thread_position_in_grid]]";
        // let position_params = "uint3 group_id [[threadgroup_position_in_grid]], uint3 local_id [[thread_position_in_threadgroup]]";

        Ok(format!(
            "kernel void shift_and_dist({input_output_type}, {position_params})"
        ))
    }

    /// Generates the shader body to perform pattern matching
    fn generate_shader_body(
        automaton: &ShiftAndDistAutomaton,
        state_type: &str,
        first_char_index: usize,
        last_char_index: usize,
        count_matches_per_line: bool,
        vars: ShaderVariables,
    ) -> Result<String> {
        let mut body = String::with_capacity(1024);

        // body.push_str("    const uint block_no = group_id.x * THREADGROUP_SIZE * THREADGROUP_SIZE + local_id.y * THREADGROUP_SIZE + local_id.x;\n");

        // Initialize input pointer based on configuration
        if vars.use_argument_buffer {
            body.push_str(
                "    const device uint8_t* input = args.input + (position.x * BLOCK_SIZE);\n",
            );
        } else {
            body.push_str("    input = input + (position.x * BLOCK_SIZE);\n");
            // body.push_str("    input = input + (block_no * BLOCK_SIZE);\n");
        }

        // Initialize counters and state
        if count_matches_per_line {
            body.push_str("    uint matches = 0;\n");
        }

        writeln!(body, "    {state_type} state = {};", automaton.mask_start)?;
        body.push_str("    for (uint i = 0; i < BLOCK_SIZE; i++) {\n");
        body.push_str("        uint c = input[i];\n");

        // Generate the main state transition logic
        writeln!(
            body,
            "        state = (c < {first_char_index} || {last_char_index} < c) ? 0 :"
        )?;

        // Add the distance calculation code (indented)
        let distance_code = Self::generate_distance_gpu_code(automaton)?;
        for line in distance_code.lines() {
            writeln!(body, "            {}", line)?;
        }

        // Get the appropriate buffer prefixes based on features
        let buffer_prefix = if vars.use_argument_buffer {
            "args."
        } else {
            ""
        };

        let buffer_index = "position.x";
        // let buffer_index = "block_no";

        // Add match detection and output logic
        if count_matches_per_line {
            writeln!(
                body,
                "        if ((state & 0x{:x}) != 0) {{ matches += 1; }}",
                automaton.mask_final
            )?;
            body.push_str("        if (c == 0x17) { break; }\n    }\n");
            writeln!(body, "    {buffer_prefix}output[{buffer_index}] = matches;")?;
        } else {
            writeln!(body, "        if (state & 0x{:x}) {{", automaton.mask_final)?;
            writeln!(
                body,
                "            {buffer_prefix}output[{buffer_index}] = 1;\n            return;\n        }}"
            )?;
            body.push_str("        if (c == 0x17) { break; }\n    }\n");
        }

        Ok(body)
    }

    /// Compiles a Metal shader from a ShiftAndDistAutomaton
    #[instrument]
    pub fn compile_shader_from_automaton(
        device: &Id<dyn MTLDevice>,
        automaton: &ShiftAndDistAutomaton,
        count_matches_per_line: bool,
        vars: ShaderVariables,
    ) -> Result<Id<dyn MTLLibrary>> {
        // Determine state type and character range
        let state_type = Self::determine_state_type(automaton);
        let (first_char_index, last_char_index) = Self::find_char_range(automaton);
        let output_type = if count_matches_per_line {
            "uint32_t"
        } else {
            "uint8_t"
        };

        // Build the shader in multiple parts for better organization
        let mut shader = String::with_capacity(2048);

        // 1. Header section
        let header = Self::generate_shader_header(
            automaton,
            state_type,
            first_char_index,
            last_char_index,
            output_type,
            vars,
        )?;
        shader.push_str(&header);

        // 2. Kernel function signature
        let signature = Self::generate_kernel_signature(output_type, vars)?;
        shader.push_str(&signature);
        shader.push_str(" {\n");

        // 3. Kernel body
        let body = Self::generate_shader_body(
            automaton,
            state_type,
            first_char_index,
            last_char_index,
            count_matches_per_line,
            vars,
        )?;
        shader.push_str(&body);

        // Close the function
        shader.push_str("}\n");

        // Print the shader (could be replaced with debug logging)
        println!("Generated shader:\n{}", shader);

        // Compile the shader
        let shader_source = NSString::from_str(&shader);
        let compile_options = MTLCompileOptions::new();

        unsafe {
            compile_options.setOptimizationLevel(MTLLibraryOptimizationLevel::Default);
        }

        let library = device
            .newLibraryWithSource_options_error(&shader_source, Some(&compile_options))
            .map_err(|e| anyhow!(e.localizedDescription().to_string()))
            .context("Failed to compile Metal shader")?;

        Ok(library)
    }
}

impl<S: MetalStrategy> ShiftAndDist for ShiftAndDistGPU<S> {
    fn count_matches(&mut self, text: &str) -> usize {
        let mut input = text.bytes().collect::<Vec<u8>>();
        input.push(0x17);
        let input = input.into_boxed_slice();

        // We do a small trick here, setting the output buffer to 4 * u8 to get the 32bits
        // Call strategy method directly using S:: notation instead of using a field
        let (command_buffer, compute_command_encoder, output_buffer) =
            S::setup_compute_command_encoder(
                &self.device,
                &self.command_queue,
                &self.compute_pipeline_state,
                input.as_ptr(),
                input.len(),
                4,
                self.icb.as_ref(),
            )
            .unwrap();

        // Call strategy method directly using S:: notation
        S::dispatch_command_encoder(
            &self.compute_pipeline_state,
            &compute_command_encoder,
            1,
            self.icb.as_ref(),
        );

        compute_command_encoder.endEncoding();
        command_buffer.commit();

        unsafe { command_buffer.waitUntilCompleted() };

        let output_ptr = output_buffer.contents().as_ptr() as *const u32;

        let output = unsafe { std::slice::from_raw_parts(output_ptr, 1) }[0] as usize;

        output
    }

    #[instrument]
    fn  count_match_lines(&self, lines: &[[u8; BLOCK_SIZE]]) -> Result<usize> {
        let input_len = lines.len();
        assert!(input_len <= MAX_INPUT_SIZE, "Too many lines");

        // Call strategy method directly using S:: notation
        let (command_buffer, compute_command_encoder, output_buffer) =
            S::setup_compute_command_encoder(
                &self.device,
                &self.command_queue,
                &self.compute_pipeline_state,
                lines.as_ptr().cast::<u8>(),
                input_len * BLOCK_SIZE,
                input_len,
                self.icb.as_ref(),
            )
            .unwrap();

        let execution_span = info_span!("gpu_execution").entered();

        // Call strategy method directly using S:: notation
        S::dispatch_command_encoder(
            &self.compute_pipeline_state,
            &compute_command_encoder,
            input_len,
            self.icb.as_ref(),
        );

        compute_command_encoder.endEncoding();
        command_buffer.commit();

        unsafe { command_buffer.waitUntilCompleted() };

        execution_span.exit();

        // Performance metrics
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

        let result_processing_span = info_span!("result_processing").entered();
        let output_ptr = output_buffer.contents().as_ptr() as *const u8;
        let output: Vec<u8> = unsafe { std::slice::from_raw_parts(output_ptr, input_len).to_vec() };
        let count = output.into_iter().filter(|c| *c > 0).count();
        info!("matched_lines = {}", count);
        result_processing_span.exit();

        Ok(count)
    }

    #[instrument]
    fn match_lines_future(
        &self,
        lines: &[[u8; BLOCK_SIZE]],
        chunks: usize,
    ) -> Result<impl Future<Output = Vec<bool>>> {
        let input_len = lines.len();
        assert!(input_len <= MAX_INPUT_SIZE, "Too many lines");

        let (command_buffer, compute_command_encoder, output_buffer) =
            S::setup_compute_command_encoder(
                &self.device,
                &self.command_queue,
                &self.compute_pipeline_state,
                lines.as_ptr().cast::<u8>(),
                input_len * BLOCK_SIZE,
                input_len,
                self.icb.as_ref(),
            )
            .unwrap();

        let execution_span = info_span!("gpu_execution").entered();

        S::dispatch_command_encoder(
            &self.compute_pipeline_state,
            &compute_command_encoder,
            input_len,
            self.icb.as_ref(),
        );

        compute_command_encoder.endEncoding();

        let shared_data = Arc::new(Mutex::new(ShiftAndFutureData {
            result: None,
            waker: None,
            output_buffer: Some(output_buffer),
        }));

        let shared_data_clone = shared_data.clone();

        let handler = RcBlock::new(
            move |command_buffer_ptr: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                // let command_buffer = unsafe { command_buffer_ptr.as_ref() };

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
