use ash::{
    Entry,
    vk::{self, ComputePipelineCreateInfo},
};
use clap::Parser;
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

    let layer_names = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
    let instance_create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_names);

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
    let extension_names = [
        c"VK_KHR_cooperative_matrix".as_ptr(),
        // vk::KhrCooperativeMatrixFn::name().as_ptr(),
        // CStr::from_bytes_with_nul(b"VK_EXT_shader_float8\0")?.as_ptr(),
        // CStr::from_bytes_with_nul(b"VK_KHR_vulkan_memory_model\0")?.as_ptr(),
    ];

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
    let shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/conv3x3.comp.spv"));
    println!("Shader bytecode length: {}", shader_code.len());

    //let shader_module_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    //let shader_module = unsafe { device.create_shader_module(shader_module_info, None) };

    //pipe_create_info = vk::ComputePipelineCreateInfo::default()
    //    .device
    //    .create_compute_pipelines(None, &[pipe_create_info], None);

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

        device.destroy_command_pool(command_pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    Ok(())
}
