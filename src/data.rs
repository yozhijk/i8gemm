use ash::{Device, vk};
use rand::distributions::{Distribution, Standard};
use rand::{self, Rng};
use std::ptr;

use super::device_buffer::DeviceBuffer;
use crate::device_buffer::BufferUsage;

pub fn create_tensor<T: Copy>(
    device: &Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    shape: &[usize],
) -> DeviceBuffer {
    let size_in_bytes = shape.iter().product::<usize>() * std::mem::size_of::<T>();

    println!(
        "Allocating tensor: {:.1}Mb",
        size_in_bytes as f32 / (1024.0 * 1024.0)
    );

    unsafe { DeviceBuffer::new(device, size_in_bytes, mem_props, BufferUsage::DeviceLocal) }
}

pub fn copy_host_to_device<T: Copy>(
    device: &Device,
    src: &[T],
    dst: &DeviceBuffer,
    command_pool: vk::CommandPool,
    command_queue: vk::Queue,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
) -> Result<(), ()> {
    unsafe {
        let size_in_bytes = std::mem::size_of_val(src);
        // Allocate staging buffer
        let staging_buffer =
            DeviceBuffer::new(device, size_in_bytes, mem_props, BufferUsage::Staging);

        // Copy data into Staging Buffer
        let ptr = device
            .map_memory(
                staging_buffer.memory,
                0,
                size_in_bytes as u64,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();

        ptr::copy_nonoverlapping(src.as_ptr(), ptr as *mut T, src.len());
        device.unmap_memory(staging_buffer.memory);

        // --- 3. Record Copy Command ---
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

        let copy_region = vk::BufferCopy::default().size(size_in_bytes as u64);
        device.cmd_copy_buffer(
            command_buffer,
            staging_buffer.buffer,
            dst.buffer,
            &[copy_region],
        );

        device.end_command_buffer(command_buffer).unwrap();

        // --- 4. Submit and Wait ---
        let cmd_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
        let queue = command_queue;
        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .unwrap();
        device.queue_wait_idle(queue).unwrap(); // Wait for copy to finish
        staging_buffer.destroy(device);
    }

    Ok(())
}

pub fn copy_device_to_host<T: Copy>(
    device: &Device,
    src: &DeviceBuffer,
    dst: &mut [T],
    command_pool: vk::CommandPool,
    command_queue: vk::Queue,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
) -> Result<(), ()> {
    unsafe {
        let size_in_bytes = std::mem::size_of_val(dst);
        // Allocate staging buffer
        let staging_buffer =
            DeviceBuffer::new(device, size_in_bytes, mem_props, BufferUsage::Staging);

        // --- 3. Record Copy Command ---
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

        let copy_region = vk::BufferCopy::default().size(size_in_bytes as u64);
        device.cmd_copy_buffer(
            command_buffer,
            src.buffer,
            staging_buffer.buffer,
            &[copy_region],
        );

        device.end_command_buffer(command_buffer).unwrap();

        // --- 4. Submit and Wait ---
        let cmd_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
        let queue = command_queue;
        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .unwrap();
        device.queue_wait_idle(queue).unwrap(); // Wait for copy to finish

        // Copy data into Staging Buffer
        let ptr = device
            .map_memory(
                staging_buffer.memory,
                0,
                size_in_bytes as u64,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();

        ptr::copy_nonoverlapping(ptr as *mut T, dst.as_mut_ptr(), dst.len());
        device.unmap_memory(staging_buffer.memory);

        staging_buffer.destroy(device);
    }
    Ok(())
}

pub fn generate_random_data<T>(shape: &[usize]) -> Vec<T>
where
    Standard: Distribution<T>,
{
    let mut rng = rand::thread_rng();

    let count = shape.iter().product();

    let data: Vec<T> = (0..count).map(|_| rng.r#gen()).collect();
    data
}
