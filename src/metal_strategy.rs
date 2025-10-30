use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLArgumentBuffersTier, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLDispatchType,
    MTLIndirectCommandBuffer, MTLIndirectComputeCommand, MTLResource, MTLResourceOptions,
    MTLResourceUsage, MTLSize,
};
use std::fmt::Debug;
use std::mem::size_of;
use std::ptr::NonNull;
use tracing::instrument;

type Id<T> = Retained<ProtocolObject<T>>;

/// A trait that defines the strategy for executing Metal compute shaders
pub trait MetalStrategy: Debug {
    /// Creates any additional resources needed by this strategy
    fn create_resources(
        device: &Id<dyn MTLDevice>,
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
    ) -> Result<Option<(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>>;

    /// Sets up a compute command encoder
    fn setup_compute_command_encoder(
        device: &Id<dyn MTLDevice>,
        command_queue: &Id<dyn MTLCommandQueue>,
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
        input_ptr: *const u8,
        input_len: usize,
        num_blocks: usize,
        icb: Option<&(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    ) -> Result<(
        Id<dyn MTLCommandBuffer>,
        Id<dyn MTLComputeCommandEncoder>,
        Id<dyn MTLBuffer>,
    )>;

    /// Dispatches the compute command encoder
    fn dispatch_command_encoder(
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
        compute_command_encoder: &Id<dyn MTLComputeCommandEncoder>,
        input_len: usize,
        icb: Option<&(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    );

    /// Returns shader variables for configuration
    fn shader_variables() -> ShaderVariables;
}

/// Shader configuration variables based on the strategy
#[derive(Debug, Clone, Copy)]
pub struct ShaderVariables {
    pub use_argument_buffer: bool,
    pub use_indirect_command_buffer: bool,
}

/// Basic strategy without argument buffers or indirect command buffers
#[derive(Debug, Clone, Copy)]
pub struct BasicStrategy;
/// Strategy that uses argument buffers
#[derive(Debug, Clone, Copy)]
pub struct ArgumentBufferStrategy;
/// Strategy that uses indirect command buffers
#[derive(Debug, Clone, Copy)]
pub struct IndirectCommandBufferStrategy;
/// Strategy that uses both argument buffers and indirect command buffers
#[derive(Debug, Clone, Copy)]
pub struct CombinedStrategy;
/// Strategy that only performs a blit operation to test memory transfer
#[derive(Debug, Clone, Copy)]
pub struct BlitStrategy;

// Implementation for BasicStrategy
impl MetalStrategy for BasicStrategy {
    fn create_resources(
        _device: &Id<dyn MTLDevice>,
        _compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
    ) -> Result<Option<(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>> {
        Ok(None)
    }

    #[instrument]
    fn setup_compute_command_encoder(
        device: &Id<dyn MTLDevice>,
        command_queue: &Id<dyn MTLCommandQueue>,
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
        input_ptr: *const u8,
        input_len: usize,
        num_blocks: usize,
        _icb: Option<&(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    ) -> Result<(
        Id<dyn MTLCommandBuffer>,
        Id<dyn MTLComputeCommandEncoder>,
        Id<dyn MTLBuffer>,
    )> {
        let input_buffer = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
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

        let output_buffer = device
            .newBufferWithLength_options(
                std::mem::size_of::<u8>() * num_blocks,
                MTLResourceOptions::StorageModeShared,
            )
            .context("Failed to create output buffer")?;

        let command_buffer = command_queue
            .commandBuffer()
            .context("Failed to create Metal command buffer")?;

        let compute_command_encoder = command_buffer
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .context("Failed to get compute command encoder")?;

        compute_command_encoder.setComputePipelineState(&compute_pipeline_state);

        // Set buffers directly
        unsafe {
            compute_command_encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 0);
            compute_command_encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 1);
        }

        Ok((command_buffer, compute_command_encoder, output_buffer))
    }

    fn dispatch_command_encoder(
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
        compute_command_encoder: &Id<dyn MTLComputeCommandEncoder>,
        input_len: usize,
        _icb: Option<&(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    ) {
        let max = compute_pipeline_state.maxTotalThreadsPerThreadgroup();

        compute_command_encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: input_len,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: max,
                height: 1,
                depth: 1,
            },
        );

        // let num_threadgroup_threads = MTLSize {
        //     width: 32,
        //     height: 32,
        //     depth: 1,
        // };
        // let threads_per_threadgroup = (num_threadgroup_threads.width
        //     * num_threadgroup_threads.height
        //     * num_threadgroup_threads.depth) as u32;

        // let num_threadgroups = MTLSize {
        //     width: (input_len as f32 / threads_per_threadgroup as f32).ceil() as usize,
        //     height: 1,
        //     depth: 1,
        // };

        // compute_command_encoder
        //     .dispatchThreadgroups_threadsPerThreadgroup(num_threadgroups, num_threadgroup_threads);
    }

    fn shader_variables() -> ShaderVariables {
        ShaderVariables {
            use_argument_buffer: false,
            use_indirect_command_buffer: false,
        }
    }
}

// Implementation for ArgumentBufferStrategy
impl MetalStrategy for ArgumentBufferStrategy {
    fn create_resources(
        _device: &Id<dyn MTLDevice>,
        _compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
    ) -> Result<Option<(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>> {
        Ok(None)
    }

    #[instrument]
    fn setup_compute_command_encoder(
        device: &Id<dyn MTLDevice>,
        command_queue: &Id<dyn MTLCommandQueue>,
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
        input_ptr: *const u8,
        input_len: usize,
        num_blocks: usize,
        _icb: Option<&(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    ) -> Result<(
        Id<dyn MTLCommandBuffer>,
        Id<dyn MTLComputeCommandEncoder>,
        Id<dyn MTLBuffer>,
    )> {
        let input_buffer = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
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

        let output_buffer = device
            .newBufferWithLength_options(
                std::mem::size_of::<u8>() * num_blocks,
                MTLResourceOptions::StorageModeShared,
            )
            .context("Failed to create output buffer")?;

        let command_buffer = command_queue
            .commandBuffer()
            .context("Failed to create Metal command buffer")?;

        let compute_command_encoder = command_buffer
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .context("Failed to get compute command encoder")?;

        compute_command_encoder.setComputePipelineState(&compute_pipeline_state);

        // Create argument buffer
        let argument_buffer = device
            .newBufferWithLength_options(
                size_of::<u64>() * 2,
                MTLResourceOptions::StorageModeShared,
            )
            .context("Failed to create argument buffer")?;

        unsafe {
            let argument_ptr = argument_buffer.contents().cast::<u64>().as_ptr();
            let input_gpu_addr = input_buffer.gpuAddress() as u64;
            let output_gpu_addr = output_buffer.gpuAddress() as u64;

            // Write addresses to argument buffer
            argument_ptr.write(input_gpu_addr);
            argument_ptr.add(1).write(output_gpu_addr);
        }

        // Register resources with the command encoder
        compute_command_encoder.useResource_usage(
            &ProtocolObject::<dyn MTLResource>::from_ref::<ProtocolObject<dyn MTLBuffer>>(
                input_buffer.as_ref(),
            ),
            MTLResourceUsage::Read,
        );
        compute_command_encoder.useResource_usage(
            &ProtocolObject::<dyn MTLResource>::from_ref::<ProtocolObject<dyn MTLBuffer>>(
                output_buffer.as_ref(),
            ),
            MTLResourceUsage::Write,
        );

        // Set argument buffer
        unsafe {
            compute_command_encoder.setBuffer_offset_atIndex(Some(&argument_buffer), 0, 0);
        }

        Ok((command_buffer, compute_command_encoder, output_buffer))
    }

    fn dispatch_command_encoder(
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
        compute_command_encoder: &Id<dyn MTLComputeCommandEncoder>,
        input_len: usize,
        _icb: Option<&(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    ) {
        let max = compute_pipeline_state.maxTotalThreadsPerThreadgroup();

        compute_command_encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: input_len,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: max,
                height: 1,
                depth: 1,
            },
        );
    }

    fn shader_variables() -> ShaderVariables {
        ShaderVariables {
            use_argument_buffer: true,
            use_indirect_command_buffer: false,
        }
    }
}

impl MetalStrategy for IndirectCommandBufferStrategy {
    fn create_resources(
        device: &Id<dyn MTLDevice>,
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
    ) -> Result<Option<(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>> {
        use objc2_metal::{MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType};

        // Create indirect command buffer
        let icb_desc = unsafe { MTLIndirectCommandBufferDescriptor::new() };
        icb_desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatchThreads);
        icb_desc.setMaxKernelBufferBindCount(2);
        icb_desc.setMaxFragmentBufferBindCount(0);
        icb_desc.setMaxVertexBufferBindCount(0);
        icb_desc.setInheritBuffers(true);
        icb_desc.setInheritPipelineState(false);

        let icb = unsafe {
            device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &icb_desc,
                1,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .context("Failed to create indirect command buffer")?;

        // Configure the ICB with the compute pipeline state
        let icb_command = unsafe { icb.indirectComputeCommandAtIndex(0) };
        unsafe {
            icb_command.setComputePipelineState(&compute_pipeline_state);
        }

        let input_buffer = device
            .newBufferWithLength_options(
                std::mem::size_of::<u8>() * usize::pow(2, 18),
                MTLResourceOptions::StorageModeShared,
            )
            .context("Failed to create input buffer")?;

        Ok(Some((icb, input_buffer)))
    }

    #[instrument]
    fn setup_compute_command_encoder(
        device: &Id<dyn MTLDevice>,
        command_queue: &Id<dyn MTLCommandQueue>,
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
        input_ptr: *const u8,
        input_len: usize,
        num_blocks: usize,
        icb: Option<&(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    ) -> Result<(
        Id<dyn MTLCommandBuffer>,
        Id<dyn MTLComputeCommandEncoder>,
        Id<dyn MTLBuffer>,
    )> {
        // let input_buffer = unsafe {
        //     device.newBufferWithBytesNoCopy_length_options_deallocator(
        //         NonNull::new(input_ptr as *mut std::ffi::c_void)
        //             .context("Failed to create input pointer")?,
        //         size_of::<u8>() * input_len,
        //         MTLResourceOptions::StorageModeShared
        //             | MTLResourceOptions::CPUCacheModeWriteCombined
        //             | MTLResourceOptions::HazardTrackingModeUntracked,
        //         None,
        //     )
        // }
        // .context("Failed to create input buffer")?;

        let (_, input_buffer) = icb.context("Failed to get indirect command buffer")?;

        // println!("first 10 bytes of input buffer: {:#?}", unsafe {
        //     std::slice::from_raw_parts(input_buffer.contents().cast::<u8>().as_ptr(), 10)
        // });

        // println!("first 10 bytes of input ptr: {:#?}", unsafe {
        //     std::slice::from_raw_parts(input_ptr, 10)
        // });

        // println!("input len: {}", input_len);

        // memcopy input data to input buffer
        unsafe {
            let input_buffer_ptr = input_buffer.contents().cast::<u8>().as_ptr();
            println!("Copying input data to input buffer");
            println!("Input buffer ptr: {:?}", input_buffer_ptr);
            println!("Input ptr: {:?}", input_ptr);
            std::ptr::copy(input_ptr, input_buffer_ptr, 20);
        }

        println!("done copying");
        // println!("first 10 bytes of input buffer: {:#?}", unsafe {
        //     std::slice::from_raw_parts(input_buffer.contents().cast::<u8>().as_ptr(), 10)
        // });

        let output_buffer = device
            .newBufferWithLength_options(
                std::mem::size_of::<u8>() * num_blocks,
                MTLResourceOptions::StorageModeShared,
            )
            .context("Failed to create output buffer")?;

        let command_buffer = command_queue
            .commandBuffer()
            .context("Failed to create Metal command buffer")?;

        let compute_command_encoder = command_buffer
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .context("Failed to get compute command encoder")?;

        // No need to set pipeline state, it will be set from the ICB

        // Set the buffers for the ICB command
        // if let Some(icb) = icb {
        //     let icb_command = unsafe { icb.indirectComputeCommandAtIndex(0) };
        //     // unsafe {
        //     //     icb_command.setKernelBuffer_offset_atIndex(&input_buffer, 0, 0);
        //     //     icb_command.setKernelBuffer_offset_atIndex(&output_buffer, 0, 1);
        //     // }
        // }

        unsafe {
            compute_command_encoder.setBuffer_offset_atIndex(Some(input_buffer), 0, 0);
            compute_command_encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 1);
        }

        Ok((command_buffer, compute_command_encoder, output_buffer))
    }

    fn dispatch_command_encoder(
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
        compute_command_encoder: &Id<dyn MTLComputeCommandEncoder>,
        input_len: usize,
        icb: Option<&(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    ) {
        use objc2_foundation::NSRange;

        let max = compute_pipeline_state.maxTotalThreadsPerThreadgroup();

        if let Some((icb, _)) = icb {
            let dynamic_command = unsafe { icb.indirectComputeCommandAtIndex(0) };

            unsafe {
                dynamic_command.concurrentDispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: input_len,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: max,
                        height: 1,
                        depth: 1,
                    },
                );
            }

            unsafe {
                compute_command_encoder.executeCommandsInBuffer_withRange(&icb, NSRange::new(0, 1));
            }
        }
    }

    fn shader_variables() -> ShaderVariables {
        ShaderVariables {
            use_argument_buffer: false,
            use_indirect_command_buffer: true,
        }
    }
}

// Implementation for CombinedStrategy
impl MetalStrategy for CombinedStrategy {
    fn create_resources(
        device: &Id<dyn MTLDevice>,
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
    ) -> Result<Option<(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>> {
        use objc2_metal::{MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType};

        // Create indirect command buffer
        let icb_desc = unsafe { MTLIndirectCommandBufferDescriptor::new() };
        icb_desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatchThreads);
        icb_desc.setMaxKernelBufferBindCount(1); // 1 for argument buffer
        icb_desc.setMaxFragmentBufferBindCount(0);
        icb_desc.setMaxVertexBufferBindCount(0);
        icb_desc.setInheritBuffers(true);
        icb_desc.setInheritPipelineState(false);

        let icb = unsafe {
            device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &icb_desc,
                1,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .context("Failed to create indirect command buffer")?;

        // Configure the ICB with the compute pipeline state
        let icb_command = unsafe { icb.indirectComputeCommandAtIndex(0) };
        unsafe {
            icb_command.setComputePipelineState(&compute_pipeline_state);
        }

        let input_buffer = device
            .newBufferWithLength_options(
                std::mem::size_of::<u8>() * 1,
                MTLResourceOptions::StorageModeShared,
            )
            .context("Failed to create input buffer")?;

        Ok(Some((icb, input_buffer)))
    }

    #[instrument]
    fn setup_compute_command_encoder(
        device: &Id<dyn MTLDevice>,
        command_queue: &Id<dyn MTLCommandQueue>,
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
        input_ptr: *const u8,
        input_len: usize,
        num_blocks: usize,
        icb: Option<&(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    ) -> Result<(
        Id<dyn MTLCommandBuffer>,
        Id<dyn MTLComputeCommandEncoder>,
        Id<dyn MTLBuffer>,
    )> {
        let input_buffer = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
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

        let output_buffer = device
            .newBufferWithLength_options(
                std::mem::size_of::<u8>() * num_blocks,
                MTLResourceOptions::StorageModeShared,
            )
            .context("Failed to create output buffer")?;

        let command_buffer = command_queue
            .commandBuffer()
            .context("Failed to create Metal command buffer")?;

        let compute_command_encoder = command_buffer
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .context("Failed to get compute command encoder")?;

        // Create argument buffer
        let argument_buffer = device
            .newBufferWithLength_options(
                size_of::<u64>() * 2,
                MTLResourceOptions::StorageModeShared,
            )
            .context("Failed to create argument buffer")?;

        unsafe {
            let argument_ptr = argument_buffer.contents().cast::<u64>().as_ptr();
            let input_gpu_addr = input_buffer.gpuAddress() as u64;
            let output_gpu_addr = output_buffer.gpuAddress() as u64;

            // Write addresses to argument buffer
            argument_ptr.write(input_gpu_addr);
            argument_ptr.add(1).write(output_gpu_addr);
        }

        // Register resources with the command encoder
        compute_command_encoder.useResource_usage(
            &ProtocolObject::<dyn MTLResource>::from_ref::<ProtocolObject<dyn MTLBuffer>>(
                input_buffer.as_ref(),
            ),
            MTLResourceUsage::Read,
        );
        compute_command_encoder.useResource_usage(
            &ProtocolObject::<dyn MTLResource>::from_ref::<ProtocolObject<dyn MTLBuffer>>(
                output_buffer.as_ref(),
            ),
            MTLResourceUsage::Write,
        );

        // Set the argument buffer for the ICB command
        // if let Some(icb) = icb {
        // let icb_command = unsafe { icb.indirectComputeCommandAtIndex(0) };
        // unsafe {
        //     icb_command.setKernelBuffer_offset_atIndex(&argument_buffer, 0, 0);
        // }
        // }

        unsafe {
            compute_command_encoder.setBuffer_offset_atIndex(Some(&argument_buffer), 0, 0);
        }

        Ok((command_buffer, compute_command_encoder, output_buffer))
    }

    fn dispatch_command_encoder(
        compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
        compute_command_encoder: &Id<dyn MTLComputeCommandEncoder>,
        input_len: usize,
        icb: Option<&(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    ) {
        use objc2_foundation::NSRange;

        let max = compute_pipeline_state.maxTotalThreadsPerThreadgroup();

        if let Some((icb, _)) = icb {
            let dynamic_command = unsafe { icb.indirectComputeCommandAtIndex(0) };

            unsafe {
                dynamic_command.concurrentDispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: input_len,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: max,
                        height: 1,
                        depth: 1,
                    },
                );
            }

            unsafe {
                compute_command_encoder.executeCommandsInBuffer_withRange(&icb, NSRange::new(0, 1));
            }
        }
    }

    fn shader_variables() -> ShaderVariables {
        ShaderVariables {
            use_argument_buffer: true,
            use_indirect_command_buffer: true,
        }
    }
}

impl MetalStrategy for BlitStrategy {
    fn create_resources(
        _device: &Id<dyn MTLDevice>,
        _compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
    ) -> Result<Option<(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>> {
        // No ICB needed for a simple blit operation
        Ok(None)
    }

    #[instrument]
    fn setup_compute_command_encoder(
        device: &Id<dyn MTLDevice>,
        command_queue: &Id<dyn MTLCommandQueue>,
        _compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
        input_ptr: *const u8,
        input_len: usize,
        num_blocks: usize,
        _icb: Option<&(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    ) -> Result<(
        Id<dyn MTLCommandBuffer>,
        Id<dyn MTLComputeCommandEncoder>,
        Id<dyn MTLBuffer>,
    )> {
        // Create input buffer with proper options for optimal transfer
        let input_buffer = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
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

        // Create destination buffer - output buffer will be the result of the blit
        let output_buffer = device
            .newBufferWithLength_options(
                std::mem::size_of::<u8>() * input_len, // Use same size as input for complete blit
                MTLResourceOptions::StorageModePrivate, // Use private storage for optimal GPU performance
            )
            .context("Failed to create output buffer")?;

        // Create command buffer
        let command_buffer = command_queue
            .commandBuffer()
            .context("Failed to create Metal command buffer")?;

        // Create a blit command encoder
        let blit_command_encoder = command_buffer
            .blitCommandEncoder()
            .context("Failed to get blit command encoder")?;

        // Copy from input buffer to output buffer
        unsafe {
            blit_command_encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &input_buffer,
                0,
                &output_buffer,
                0,
                size_of::<u8>() * input_len,
            );
        }

        // End the blit encoding
        blit_command_encoder.endEncoding();

        // Create an empty compute encoder just to satisfy the trait requirements
        let compute_command_encoder = command_buffer
            .computeCommandEncoder()
            .context("Failed to get compute command encoder")?;

        // Return the command buffer, compute command encoder (empty), and output buffer
        Ok((command_buffer, compute_command_encoder, output_buffer))
    }

    fn dispatch_command_encoder(
        _compute_pipeline_state: &Id<dyn MTLComputePipelineState>,
        compute_command_encoder: &Id<dyn MTLComputeCommandEncoder>,
        _input_len: usize,
        _icb: Option<&(Id<dyn MTLIndirectCommandBuffer>, Id<dyn MTLBuffer>)>,
    ) {
        // End the compute encoder immediately since we're not using it
        // All the work was done in the blit encoder
        compute_command_encoder.endEncoding();
    }

    fn shader_variables() -> ShaderVariables {
        ShaderVariables {
            use_argument_buffer: false,
            use_indirect_command_buffer: false,
        }
    }
}
