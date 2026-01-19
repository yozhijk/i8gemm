use ash::{Device, vk};

#[derive(Debug, Copy, Clone)]
pub enum BufferUsage {
    Staging,
    DeviceLocal,
}

pub struct DeviceBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size_bytes: usize,
}

fn find_memorytype_index(
    mem_req: &vk::MemoryRequirements,
    mem_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    mem_prop
        .memory_types
        .iter()
        .enumerate()
        .find(|(index, mem_type)| {
            (1 << index) & mem_req.memory_type_bits != 0 && mem_type.property_flags.contains(flags)
        })
        .map(|(index, _)| index as u32)
}

fn usage_to_flags(usage: BufferUsage) -> vk::BufferUsageFlags {
    match usage {
        BufferUsage::DeviceLocal => {
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::STORAGE_BUFFER
        }
        BufferUsage::Staging => {
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST
        }
    }
}

fn usage_to_memflags(usage: BufferUsage) -> vk::MemoryPropertyFlags {
    match usage {
        BufferUsage::DeviceLocal => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        BufferUsage::Staging => vk::MemoryPropertyFlags::HOST_VISIBLE,
    }
}

impl DeviceBuffer {
    pub unsafe fn new(
        device: &Device,
        size: usize,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        usage: BufferUsage,
    ) -> Self {
        // --- 1. Create Device Local Buffer (GPU Only) ---
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage_to_flags(usage))
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device.create_buffer(&buffer_create_info, None).unwrap();
        let buffer_req = device.get_buffer_memory_requirements(buffer);

        let mem_index = find_memorytype_index(
            &buffer_req,
            &mem_props,
            usage_to_memflags(usage), // Fast GPU access
        )
        .expect("No Device Local memory found");

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(buffer_req.size)
            .memory_type_index(mem_index);
        let memory = device.allocate_memory(&alloc_info, None).unwrap();
        let _ = device.bind_buffer_memory(buffer, memory, 0);

        DeviceBuffer {
            buffer,
            memory,
            size_bytes: size,
        }
    }

    pub unsafe fn destroy(self, device: &Device) {
        device.destroy_buffer(self.buffer, None);
        device.free_memory(self.memory, None);
    }
}
