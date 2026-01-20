use ash::util::read_spv;
use ash::vk::DescriptorSetLayout;
use ash::{
    Device, Entry, Instance,
    vk::{self},
};
use clap::Parser;
//use renderdoc::RenderDoc;
use std::io::Cursor;

mod data;
mod device_buffer;

use data::create_tensor;

use crate::data::{copy_host_to_device, generate_random_data};
use crate::device_buffer::DeviceBuffer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None, disable_help_flag = true)]
struct Args {
    /// Tensor input channels
    #[arg(short, long, default_value_t = 32)]
    in_channels: usize,

    /// Tensor output channels
    #[arg(short, long, default_value_t = 32)]
    out_channels: usize,

    /// Tensor width
    #[arg(short, long, default_value_t = 1024)]
    width: usize,

    /// Tensor height
    #[arg(short, long, default_value_t = 1024)]
    height: usize,

    /// Number of tests
    #[arg(short, long, default_value_t = 100)]
    tests: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PushConstants {
    pub num_ic: u32,
    pub num_oc: u32,
    pub height: u32,
    pub width: u32,
    pub pad: [u32; 4],
}

pub struct VulkanApi {
    pub entry: Entry,
    pub instance: Instance,
    pub pdevice: vk::PhysicalDevice,
    pub device: Device,
    pub queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub desc_pool: vk::DescriptorPool,
    pub query_pool: vk::QueryPool,
    pub props: vk::PhysicalDeviceProperties,
}

fn get_required_layers() -> Vec<*const i8> {
    vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
}

fn get_required_instance_extensions() -> Vec<*const i8> {
    if cfg!(target_os = "macos") {
        vec![c"VK_KHR_portability_enumeration".as_ptr()]
    } else {
        vec![]
    }
}

fn get_required_device_extensions() -> Vec<*const i8> {
    if cfg!(target_os = "macos") {
        vec![c"VK_KHR_portability_subset".as_ptr()]
    } else {
        //vec![c"VK_KHR_cooperative_matrix".as_ptr()]
        vec![]
    }
}

fn get_required_instance_flags() -> vk::InstanceCreateFlags {
    if cfg!(target_os = "macos") {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    }
}

fn create_vulkan_api() -> VulkanApi {
    unsafe {
        // Load Vulkan Entry
        // (This loads the Vulkan dynamic library from the system)
        let entry = Entry::load().expect("Cannot create Vulkan entry");

        // Create Instance
        // We request Vulkan 1.3 because of Cooperative Matrix
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);

        let layer_names = get_required_layers();
        let layer_extension_names = get_required_instance_extensions();
        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&layer_extension_names)
            .flags(get_required_instance_flags());

        let instance = entry
            .create_instance(&instance_create_info, None)
            .expect("Cannot create Vulkan instance");

        // Pick Physical Device (GPU), just pick the first one
        let pdevices = instance
            .enumerate_physical_devices()
            .expect("Cannot enumerate physical devices");
        let pdevice = pdevices.first().expect("No Vulkan physical device found!");
        let queue_family_properties =
            instance.get_physical_device_queue_family_properties(*pdevice);

        let queue_family_index = queue_family_properties
            .iter()
            .enumerate()
            .find(|(_, info)| info.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(index, _)| index as u32)
            .expect("No Compute Queue found!");

        let queue_priorities = [1.0];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        // --- FP8 GEMM SPECIFIC SETUP START ---
        let extension_names = get_required_device_extensions();

        // You will also need to chain specific feature structs here (p_next)
        // e.g., vk::PhysicalDeviceCooperativeMatrixFeaturesKHR
        // let mut features13 = vk::PhysicalDeviceVulkan13Features::default().compute_full_subgroups(true);
        // --- FP8 GEMM SPECIFIC SETUP END ---

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&extension_names);
        //        .push_next(&mut features13); // Example of enabling Vulkan 1.3 features

        // Create logical device
        let device = instance
            .create_device(*pdevice, &device_create_info, None)
            .expect("Cannot create Vulkan device");
        let queue = device.get_device_queue(queue_family_index, 0);

        let pool_create_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
        let command_pool = device
            .create_command_pool(&pool_create_info, None)
            .expect("Cannot create command pool");
        let mem_props = instance.get_physical_device_memory_properties(*pdevice);

        // Create descriptor pool for our buffers
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(128)];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let desc_pool = device
            .create_descriptor_pool(&pool_info, None)
            .expect("Failed to create descriptor pool");

        // 1. Get Timestamp Period (ns per tick)
        let props = instance.get_physical_device_properties(*pdevice);

        // 2. Create Query Pool for 2 timestamps (Start & End)
        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2); // Index 0 = Start, Index 1 = End

        let query_pool = device
            .create_query_pool(&create_info, None)
            .expect("Failed to create query pool");

        VulkanApi {
            entry,
            instance,
            pdevice: *pdevice,
            device,
            queue,
            command_pool,
            mem_props,
            desc_pool,
            query_pool,
            props,
        }
    }
}

fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    // Define the 3 bindings corresponding to your shader:
    // layout(binding = 0) buffer Input { ... }
    // layout(binding = 1) buffer Weight { ... }
    // layout(binding = 2) buffer Output { ... }
    unsafe {
        let bindings = [
            // Binding 0: Input tensor
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 1: Weights tensor
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 2: Output tensor
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        device
            .create_descriptor_set_layout(&layout_info, None)
            .expect("Failed to create descriptor set layout")
    }
}

fn create_compute_pipeline(
    device: &Device,
    desc_set_layout: &DescriptorSetLayout,
) -> (vk::Pipeline, vk::PipelineLayout) {
    unsafe {
        // Load SPIRV
        // Note: This path is relative to the file where this macro is called (src/main.rs)
        let shader_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/conv3x3.comp.spv"));
        let shader_code = read_spv(&mut Cursor::new(shader_bytes)).unwrap();

        // Create shader module
        let shader_module_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let shader_module = device
            .create_shader_module(&shader_module_info, None)
            .expect("Error creating shader module");

        // Create pipeline layout
        let desc_set_layouts = [*desc_set_layout];
        let push_constants_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&desc_set_layouts)
            .push_constant_ranges(&push_constants_ranges);
        let pipeline_layout = device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .expect("Cannot create pipeline layout");
        let stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        let pipeline_create_info = vk::ComputePipelineCreateInfo::default()
            .layout(pipeline_layout)
            .stage(stage_create_info);
        let pipeline = device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            .expect("Failed to create compute pipeline")[0];

        device.destroy_shader_module(shader_module, None);
        (pipeline, pipeline_layout)
    }
}

fn create_descriptor_set(
    device: &Device,
    layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    t_input: &DeviceBuffer,
    t_weight: &DeviceBuffer,
    t_output: &DeviceBuffer,
) -> vk::DescriptorSet {
    unsafe {
        let desc_set_layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&desc_set_layouts);

        let descriptor_sets = device
            .allocate_descriptor_sets(&alloc_info)
            .expect("Failed to allocate descriptor sets");

        let descriptor_set = descriptor_sets[0];

        let info_input = [vk::DescriptorBufferInfo::default()
            .buffer(t_input.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let info_weight = [vk::DescriptorBufferInfo::default()
            .buffer(t_weight.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let info_output = [vk::DescriptorBufferInfo::default()
            .buffer(t_output.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let write_sets = [
            // Binding 0: Input tensor
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info_input),
            // Binding 1: Weight tensor
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info_weight),
            // Binding 2: Output tensor
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info_output),
        ];

        device.update_descriptor_sets(&write_sets, &[]);
        descriptor_set
    }
}

pub fn get_execution_time_ns(
    device: &Device,
    query_pool: vk::QueryPool,
    timestamp_period: f32,
) -> f64 {
    unsafe {
        // Array to hold the two 64-bit timestamps
        let mut timestamps = [0u64; 2];

        // Fetch results from the GPU
        // flag VK_QUERY_RESULT_64_BIT is critical to get u64 instead of u32
        // flag VK_QUERY_RESULT_WAIT ensures we don't return until data is ready
        let result = device.get_query_pool_results(
            query_pool,
            0,
            &mut timestamps, // Output buffer
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        );

        match result {
            Ok(_) => {
                let start = timestamps[0];
                let end = timestamps[1];

                // Calculate delta ticks
                let delta_ticks = end.wrapping_sub(start);

                // Convert to nanoseconds
                (delta_ticks as f64) * (timestamp_period as f64)
            }
            Err(e) => {
                eprintln!("Failed to get query results: {:?}", e);
                0.0
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = Parser::parse();

    println!("Benchmarking with:");
    println!(
        "I: {}, O: {}, W: {}, H: {}",
        args.in_channels, args.out_channels, args.width, args.height
    );
    println!("# of iterations: {}", args.tests);

    // RenderDoc
    // let mut rd: RenderDoc<renderdoc::V141> = RenderDoc::new().expect("Cannot connect to RenderDoc");
    // rd.start_frame_capture(std::ptr::null(), std::ptr::null());

    let api = create_vulkan_api();

    let device = api.device;
    let queue = api.queue;

    // Create compute pipeline
    let desc_set_layout = create_descriptor_set_layout(&device);
    let (pipeline, pipeline_layout) = create_compute_pipeline(&device, &desc_set_layout);

    //  Create device buffers
    // Input tensor
    let input_shape = [args.in_channels, args.height, args.width];
    let input_data = generate_random_data::<i8>(&input_shape);
    let t_input = create_tensor::<i8>(&device, &api.mem_props, &input_shape);
    let _ = copy_host_to_device(
        &device,
        &input_data,
        &t_input,
        api.command_pool,
        queue,
        &api.mem_props,
    );

    // Weight tensor
    let weight_shape = [args.in_channels, args.out_channels, 3, 3];
    let weight_data = generate_random_data::<i8>(&weight_shape);
    let t_weight = create_tensor::<i8>(&device, &api.mem_props, &weight_shape);
    let _ = copy_host_to_device(
        &device,
        &weight_data,
        &t_weight,
        api.command_pool,
        queue,
        &api.mem_props,
    );

    // Output tensor
    let output_shape = [args.out_channels, args.height, args.width];
    let t_output = create_tensor::<i32>(&device, &api.mem_props, &output_shape);

    // Create descriptor set for our buffers
    let desc_set = create_descriptor_set(
        &device,
        desc_set_layout,
        api.desc_pool,
        &t_input,
        &t_weight,
        &t_output,
    );

    // Run compute pass
    unsafe {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(api.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&allocate_info).unwrap()[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(command_buffer, &begin_info)
            .unwrap();

        // Bind Pipeline
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

        // Bind Descriptors
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0, // First set index
            &[desc_set],
            &[], // Dynamic offsets
        );

        let constants = PushConstants {
            num_ic: args.in_channels as u32,
            num_oc: args.out_channels as u32,
            height: args.height as u32,
            width: args.width as u32,
            pad: [42, 11, 5, 16],
        };

        // "Serialize" struct to bytes (unsafe cast)
        let constants_ptr = &constants as *const PushConstants as *const u8;
        let constants_bytes =
            std::slice::from_raw_parts(constants_ptr, std::mem::size_of::<PushConstants>());

        device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0, // offset
            constants_bytes,
        );

        device.cmd_reset_query_pool(command_buffer, api.query_pool, 0, 2);
        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            api.query_pool,
            0,
        );

        // Calculate Dispatch Dimensions
        // CAUTION: This depends entirely on your shader's local_size_x/y
        let workgroup_size = 8;
        let group_count_x = args.width.div_ceil(workgroup_size);
        let group_count_y = args.height.div_ceil(workgroup_size);

        // Dispatch
        device.cmd_dispatch(
            command_buffer,
            group_count_x as u32,
            group_count_y as u32,
            1,
        );

        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            api.query_pool,
            1,
        );

        device.end_command_buffer(command_buffer).unwrap();

        // --- 4. Submit and Wait ---
        let cmd_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .unwrap();
        device.queue_wait_idle(queue).unwrap(); // Wait for copy to finish

        let gpu_time =
            get_execution_time_ns(&device, api.query_pool, api.props.limits.timestamp_period);

        println!("Dispatch took: {:.3} ms", gpu_time / 1e6);

        device.destroy_descriptor_pool(api.desc_pool, None);
    }

    // Download data

    // Compare to golden

    // RenderDoc
    // rd.end_frame_capture(std::ptr::null(), std::ptr::null());

    // 7. Cleanup;
    unsafe {
        t_input.destroy(&device);
        t_output.destroy(&device);
        t_weight.destroy(&device);

        device.destroy_query_pool(api.query_pool, None);
        device.destroy_descriptor_set_layout(desc_set_layout, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_command_pool(api.command_pool, None);
        device.destroy_device(None);
        api.instance.destroy_instance(None);
    }

    Ok(())
}
