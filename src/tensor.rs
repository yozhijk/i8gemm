use ash::vk;

pub struct Tensor {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size_bytes: u64,
}
