use ash::util::read_spv;
use ash::vk::DescriptorSetLayout;
use ash::{
    Device, Entry, Instance,
    vk::{self},
};
use clap::Parser;
// use renderdoc::RenderDoc;
use std::io::Cursor;

mod data;
mod device_buffer;

use data::{array_exact_compare, create_tensor};

use crate::data::{
    conv3x3_i8_acc_i32, copy_device_to_host, copy_host_to_device, dump_hwc_to_csv,
    generate_random_data, reorder_weights_nchw,
};
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
    pub queue_props: vk::QueueFamilyProperties,
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
        vec![c"VK_KHR_cooperative_matrix".as_ptr()]
    }
}

fn get_required_instance_flags() -> vk::InstanceCreateFlags {
    if cfg!(target_os = "macos") {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    }
}

fn print_supported_matrix_sizes(entry: &Entry, instance: &Instance, pdevice: vk::PhysicalDevice) {
    unsafe {
        let coop_mat_fn = ash::khr::cooperative_matrix::Instance::new(entry, instance);

        let props = coop_mat_fn
            .get_physical_device_cooperative_matrix_properties(pdevice)
            .expect("Failed to query cooperative matrix properties");

        for p in props {
            if p.a_type == vk::ComponentTypeKHR::SINT8
                && p.c_type == vk::ComponentTypeKHR::SINT32
                && p.scope == vk::ScopeKHR::SUBGROUP
            {
                println!(
                    "Supported: M={} N={} K={} (A: {:?}, C: {:?})",
                    p.m_size, p.n_size, p.k_size, p.a_type, p.c_type
                );
            }
        }
    }
}

fn create_vulkan_api(num_queries: u32) -> VulkanApi {
    unsafe {
        // (This loads the Vulkan dynamic library from the system)
        let entry = Entry::load().expect("Cannot create Vulkan entry");

        // We request Vulkan 1.3 because of Cooperative Matrix.
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

        // Pick physical device, just pick the first one.
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

        let extension_names = get_required_device_extensions();

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .vulkan_memory_model(true) // Fixes Error 2
            .shader_float16(true) // Likely needed for tensor ops
            .shader_int8(true)
            .uniform_and_storage_buffer8_bit_access(true)
            .storage_buffer8_bit_access(true)
            .host_query_reset(true);
        let mut coop_matrix_features =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&extension_names)
            .push_next(&mut features12)
            .push_next(&mut coop_matrix_features);

        let device = instance
            .create_device(*pdevice, &device_create_info, None)
            .expect("Cannot create Vulkan device");
        let queue = device.get_device_queue(queue_family_index, 0);

        print_supported_matrix_sizes(&entry, &instance, *pdevice);

        let pool_create_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
        let command_pool = device
            .create_command_pool(&pool_create_info, None)
            .expect("Cannot create command pool");
        let mem_props = instance.get_physical_device_memory_properties(*pdevice);

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(128)];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let desc_pool = device
            .create_descriptor_pool(&pool_info, None)
            .expect("Failed to create descriptor pool");

        let props = instance.get_physical_device_properties(*pdevice);

        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2 * num_queries);

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
            queue_props: queue_family_properties[queue_family_index as usize],
        }
    }
}

fn destroy_vulkan_api(api: VulkanApi) {
    unsafe {
        api.device.destroy_query_pool(api.query_pool, None);
        api.device.destroy_command_pool(api.command_pool, None);
        api.device.destroy_device(None);
        api.instance.destroy_instance(None);
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
        // Load SPIRV.
        // Note: This path is relative to the file where this macro is called (src/main.rs).
        let shader_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/conv3x3.coopmat.comp.spv"));
        let shader_code = read_spv(&mut Cursor::new(shader_bytes)).unwrap();

        // Create shader module.
        let shader_module_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let shader_module = device
            .create_shader_module(&shader_module_info, None)
            .expect("Error creating shader module");

        // Create pipeline layout.
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
    index: usize,
    query_pool: vk::QueryPool,
    timestamp_period: f32,
    valid_bits: u32,
) -> f64 {
    unsafe {
        let mut timestamps = [0u64; 2];

        // Fetch results from the GPU
        // flag VK_QUERY_RESULT_64_BIT is critical to get u64 instead of u32
        // flag VK_QUERY_RESULT_WAIT ensures we don't return until data is ready
        let result = device.get_query_pool_results(
            query_pool,
            index as u32 * 2,
            &mut timestamps, // Output buffer
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        );

        match result {
            Ok(_) => {
                let start = timestamps[0];
                let end = timestamps[1];

                // Calculate valid mask
                let mask = if valid_bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << valid_bits) - 1
                };

                let delta_ticks = (end & mask).wrapping_sub(start & mask) & mask;

                (delta_ticks as f64) * (timestamp_period as f64)
            }
            Err(e) => {
                eprintln!("Failed to get query results: {:?}", e);
                0.0
            }
        }
    }
}

fn check_input_sizes(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.in_channels & 31 != 0 {
        return Err(format!(
            "ERROR: in_channels should be multiple of 32, but given {}",
            args.in_channels
        )
        .into());
    }

    if args.out_channels & 15 != 0 {
        return Err(format!(
            "ERROR: out_channels should be multiple of 16, but given {}",
            args.out_channels
        )
        .into());
    }

    if args.width & 15 != 0 {
        return Err(format!(
            "ERROR: width should be multiple of 16, but given {}",
            args.width
        )
        .into());
    }

    if args.height & 15 != 0 {
        return Err(format!(
            "ERROR: height should be multiple of 16, but given {}",
            args.height
        )
        .into());
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = Parser::parse();

    println!("Benchmarking with:");
    println!(
        "I: {}, O: {}, W: {}, H: {}",
        args.in_channels, args.out_channels, args.width, args.height
    );
    println!("# of iterations: {}", args.tests);

    check_input_sizes(&args)?;

    // RenderDoc
    // let mut rd: RenderDoc<renderdoc::V141> = RenderDoc::new().expect("Cannot connect to RenderDoc");
    // rd.start_frame_capture(std::ptr::null(), std::ptr::null());

    let api = create_vulkan_api(args.tests as u32);

    let device = &api.device;
    let queue = api.queue;

    // Create compute pipeline.
    let desc_set_layout = create_descriptor_set_layout(device);
    let (pipeline, pipeline_layout) = create_compute_pipeline(device, &desc_set_layout);

    // Create device buffers.
    // Input tensor.
    let input_shape = [args.in_channels, args.height, args.width];
    let input_data = generate_random_data::<i8>(&input_shape);
    //let input_data = vec![1_i8; args.in_channels * args.height * args.width];
    //let input_data = generate_row_number::<i8>(args.in_channels, args.height, args.width);
    let t_input = create_tensor::<i8>(device, &api.mem_props, &input_shape);
    let _ = copy_host_to_device(
        device,
        &input_data,
        &t_input,
        api.command_pool,
        queue,
        &api.mem_props,
    );

    // Weight TILE_SIZE.
    let weight_shape = [args.in_channels, args.out_channels, 3, 3];
    let weight_data = generate_random_data::<i8>(&weight_shape);
    // let weight_data = generate_copy_conv3x3_weights::<i8>(weight_shape[0]);
    // let weight_data = vec![1_i8; args.in_channels * args.out_channels * 9];
    let mut weight_reordered_data = vec![0_i8; weight_data.len()];
    reorder_weights_nchw(
        args.in_channels,
        args.out_channels,
        32,
        16,
        &weight_data,
        &mut weight_reordered_data,
    );

    let t_weight = create_tensor::<i8>(device, &api.mem_props, &weight_shape);
    let _ = copy_host_to_device(
        device,
        &weight_reordered_data,
        &t_weight,
        api.command_pool,
        queue,
        &api.mem_props,
    );

    // Output tensor.
    let output_shape = [args.out_channels, args.height, args.width];
    let t_output = create_tensor::<i32>(device, &api.mem_props, &output_shape);

    // Create descriptor set for our buffers.
    let desc_set = create_descriptor_set(
        device,
        desc_set_layout,
        api.desc_pool,
        &t_input,
        &t_weight,
        &t_output,
    );

    unsafe {
        device.reset_query_pool(api.query_pool, 0, 2 * args.tests as u32);

        for t in 0..args.tests {
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

            // Bind Pipeline.
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

            // Bind Descriptors.
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[desc_set],
                &[],
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

            device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                api.query_pool,
                2 * t as u32,
            );

            // Tile size matches shader: 32x4 spatial with 4 subgroups
            let tile_size = (32, 4);
            let group_count_x = args.width.div_ceil(tile_size.0);
            let group_count_y = args.height.div_ceil(tile_size.1);
            let group_count_z = args.out_channels / 16;

            device.cmd_dispatch(
                command_buffer,
                group_count_x as u32,
                group_count_y as u32,
                group_count_z as u32,
            );

            let output_write_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .buffer(t_output.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE);

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::default(),
                &[],
                &[output_write_barrier],
                &[],
            );

            device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                api.query_pool,
                2 * t as u32 + 1,
            );

            device.end_command_buffer(command_buffer).unwrap();

            let cmd_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
            device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .unwrap();
        }

        device.queue_wait_idle(queue).unwrap(); // Wait for copy to finish

        let mut total_exec_time = 0.0;

        for t in 0..args.tests {
            total_exec_time += get_execution_time_ns(
                device,
                t,
                api.query_pool,
                api.props.limits.timestamp_period,
                api.queue_props.timestamp_valid_bits,
            );
        }

        total_exec_time /= args.tests as f64;

        println!("Dispatch took: {:.3} ms", total_exec_time / 1e6);

        let ops = args.in_channels * args.out_channels * 9 * args.width * args.height * 2;

        println!(
            "Achieved INT8 throughput: {} tops/s",
            ops as f64 / (total_exec_time * 1e3)
        )
    }

    let num_output_elements = args.width * args.height * args.out_channels;
    let mut gpu_output = vec![0; num_output_elements];

    // Download data
    let _ = copy_device_to_host(
        device,
        &t_output,
        &mut gpu_output,
        api.command_pool,
        queue,
        &api.mem_props,
    );

    // Compare to golden
    let mut host_output = vec![0_i32; args.out_channels * args.height * args.width];
    conv3x3_i8_acc_i32(
        &input_shape,
        &output_shape,
        &input_data,
        &weight_data,
        &mut host_output,
    );

    let _ = array_exact_compare(&host_output, &gpu_output, "CPU", "GPU");

    // Print first 10 elements
    //println!("Some elements CPU: {:?}", &host_output[0..64]);
    //println!("Some elements GPU: {:?}", &gpu_output[0..64]);
    //println!("CPU elements: {:?}", &host_output);
    //println!("GPU elements: {:?}", &gpu_output);
    //println!("Some elements GPU: {:?}", &gpu_output[0..]);
    // dump_hwc_to_csv(&gpu_output, args.width, args.out_channels, "gpu_out.csv")?;
    // dump_hwc_to_csv(&host_output, args.width, args.out_channels, "cpu_out.csv")?;
    // dump_hwc_to_csv(&input_data, args.width, args.out_channels, "gpu_in.csv")?;

    // RenderDoc
    // rd.end_frame_capture(std::ptr::null(), std::ptr::null());

    unsafe {
        t_input.destroy(device);
        t_output.destroy(device);
        t_weight.destroy(device);

        device.destroy_descriptor_pool(api.desc_pool, None);
        device.destroy_descriptor_set_layout(desc_set_layout, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_pipeline(pipeline, None);
    }

    destroy_vulkan_api(api);
    Ok(())
}
