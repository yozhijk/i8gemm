use ash::{Entry, vk};
use clap::Parser;

mod tensor;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None, disable_help_flag = true)]
struct Args {
    /// Tensor input channels
    #[arg(short, long, default_value_t = 1024)]
    in_channels: usize,

    /// Tensor output channels
    #[arg(short, long, default_value_t = 1024)]
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

    println!("Vulkan Compute Device Created Successfully!");

    // 7. Load SPIRV
    //
    // Note: This path is relative to the file where this macro is called (src/main.rs)
    let shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/conv3x3.comp.spv"));
    println!("Shader bytecode length: {}", shader_code.len());

    // 7. Cleanup
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    Ok(())
}
