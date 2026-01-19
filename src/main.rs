use ash::util::read_spv;
use ash::{
    Device, Entry,
    vk::{self, ComputePipelineCreateInfo},
};
use clap::Parser;
use std::io::Cursor;
//use renderdoc::RenderDoc;

mod data;
mod device_buffer;

use data::create_tensor;

use crate::data::{copy_host_to_device, generate_random_data};

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

pub fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
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

    // println!("RenderDoc API Connected!");

    // RenderDoc
    // rd.start_frame_capture(std::ptr::null(), std::ptr::null());

    // 1. Load Vulkan Entry
    //    (This loads the Vulkan dynamic library from the system)
    let entry = unsafe { Entry::load()? };

    // 2. Create Instance
    //    We request Vulkan 1.3 because of Cooperative Matrix
    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);

    let layer_names = get_required_layers();
    let layer_extension_names = get_required_instance_extensions();
    let instance_create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_names)
        .enabled_extension_names(&layer_extension_names)
        .flags(get_required_instance_flags());

    let instance = unsafe { entry.create_instance(&instance_create_info, None)? };

    // 3. Pick Physical Device (GPU)
    //    We just pick the first one.
    let pdevices = unsafe { instance.enumerate_physical_devices()? };
    let pdevice = pdevices.first().expect("No Vulkan physical device found!");

    // 4. Find a Compute Queue Family
    let queue_family_properties =
        unsafe { instance.get_physical_device_queue_family_properties(*pdevice) };

    let queue_family_index = queue_family_properties
        .iter()
        .enumerate()
        .find(|(_, info)| info.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|(index, _)| index as u32)
        .expect("No Compute Queue found!");

    // 5. Define Device Creation Info
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

    // 6. Create Logical Device
    let device = unsafe { instance.create_device(*pdevice, &device_create_info, None)? };
    let command_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    println!("Vulkan Compute Device Created Successfully!");

    // 7. Load SPIRV
    //
    // Note: This path is relative to the file where this macro is called (src/main.rs)
    let shader_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/conv3x3.comp.spv"));
    let shader_code = read_spv(&mut Cursor::new(shader_bytes)).unwrap();
    println!("Shader bytecode length: {}", shader_code.len());

    // Create shader module
    let shader_module_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    let shader_module = unsafe {
        device
            .create_shader_module(&shader_module_info, None)
            .unwrap()
    };

    // Create descriptor set layout
    let desc_set_layout = create_descriptor_set_layout(&device);
    let desc_set_layouts = [desc_set_layout];
    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfo::default().set_layouts(&desc_set_layouts);
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .unwrap()
    };

    let stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(c"main");

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .layout(pipeline_layout)
        .stage(stage_create_info);

    let pipeline = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .expect("Failed to create compute pipeline")[0]
    };

    // 8. Create device buffers
    let pool_create_info = vk::CommandPoolCreateInfo::default();

    let command_pool = unsafe { device.create_command_pool(&pool_create_info, None).unwrap() };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(*pdevice) };
    // Input tensor
    let input_shape = [args.in_channels, args.height, args.width];
    let input_data = generate_random_data::<i8>(&input_shape);
    let t_input = create_tensor::<i8>(&device, &mem_props, &input_shape);
    let _ = copy_host_to_device(
        &device,
        &input_data,
        &t_input,
        command_pool,
        command_queue,
        &mem_props,
    );

    // Weight tensor
    let weight_shape = [args.in_channels, args.out_channels, 3, 3];
    let weight_data = generate_random_data::<i8>(&weight_shape);
    let t_weight = create_tensor::<i8>(&device, &mem_props, &weight_shape);
    let _ = copy_host_to_device(
        &device,
        &weight_data,
        &t_weight,
        command_pool,
        command_queue,
        &mem_props,
    );

    // Output tensor
    let output_shape = [args.out_channels, args.height, args.width];
    let t_output = create_tensor::<i32>(&device, &mem_props, &output_shape);

    // Run compute pass
    // --- 3. Record Copy Command ---
    unsafe {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&allocate_info).unwrap()[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(command_buffer, &begin_info)
            .unwrap();

        device.end_command_buffer(command_buffer).unwrap();

        // --- 4. Submit and Wait ---
        let cmd_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
        let queue = command_queue;
        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .unwrap();
        device.queue_wait_idle(queue).unwrap(); // Wait for copy to finish
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

        device.destroy_descriptor_set_layout(desc_set_layout, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_shader_module(shader_module, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_command_pool(command_pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    Ok(())
}
