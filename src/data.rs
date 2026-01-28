use ash::{Device, vk};
use rand::distributions::{Distribution, Standard};
use rand::{self, Rng};
use std::fmt::Display;
use std::fs::File;
use std::io::{self, BufWriter, Write};
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

pub fn conv3x3_i8_acc_i32(
    input_shape: &[usize],
    output_shape: &[usize],
    input_data: &[i8],
    weight_data: &[i8],
    output_data: &mut [i32],
) {
    let (ic, h, w) = (
        input_shape[0] as isize,
        input_shape[1] as isize,
        input_shape[2] as isize,
    );
    let oc = output_shape[0] as isize;

    let get_input_value = |i: isize, y: isize, x: isize| -> i8 {
        if x < 0 || x >= w || y < 0 || y >= h {
            0_i8
        } else {
            input_data[(y * w * ic + x * ic + i) as usize]
        }
    };

    let get_weight_value = |o: isize, i: isize, dy: isize, dx: isize| -> i8 {
        let y = dy + 1;
        let x = dx + 1;
        weight_data[(o * ic * 3 * 3 + i * 3 * 3 + y * 3 + x) as usize]
    };

    for y in 0_isize..h {
        for x in 0_isize..w {
            for o in 0_isize..oc {
                let mut res = 0_i32;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for i in 0..ic {
                            res += get_input_value(i, y + dy, x + dx) as i32
                                * get_weight_value(o, i, dy, dx) as i32;
                        }
                    }
                }
                // output_data[(o * h * w + y * w + x) as usize] = res;
                output_data[(y * w * oc + x * oc + o) as usize] = res;
            }
        }
    }
}

pub fn array_exact_compare<T: PartialEq + Display>(
    a: &[T],
    b: &[T],
    a_label: &str,
    b_label: &str,
) -> bool {
    let mismatch_idx = a.iter().zip(b).position(|(a, b)| a != b);

    if let Some(idx) = mismatch_idx {
        println!(
            "Mismatch at index {}: {}={} vs {}={}",
            idx, a_label, a[idx], b_label, b[idx]
        );
        false
    } else {
        println!("Arrays match perfectly!");
        true
    }
}

pub fn reorder_weights_nchw<T: Copy>(
    ic: usize,
    oc: usize,
    ic_slice: usize,
    oc_slice: usize,
    weights: &[T],
    reordered: &mut [T],
) {
    // Original format strides
    let oc_stride = ic * 9;

    // Sliced formate strides
    let num_input_slices = ic / ic_slice;
    let num_output_slices = oc / oc_slice;
    let matrix_size = ic_slice * oc_slice;
    let input_slice_stride = matrix_size * 9;
    let output_slice_stride = input_slice_stride * num_input_slices;

    for output_slice in 0..num_output_slices {
        for input_slice in 0..num_input_slices {
            for k in 0..9 {
                for elem in 0..matrix_size {
                    let ich = input_slice * ic_slice + elem / oc_slice;
                    let och = output_slice * oc_slice + elem % oc_slice;
                    reordered[output_slice * output_slice_stride
                        + input_slice * input_slice_stride
                        + k * matrix_size
                        + elem] = weights[och * oc_stride + ich * 9 + k];
                }
            }
        }
    }
}

pub fn dump_hwc_to_csv(data: &[i32], width: usize, channels: usize, path: &str) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write Header: row, col, ch0, ch1, ...
    write!(writer, "row,col")?;
    for c in 0..channels {
        write!(writer, ",ch{}", c)?;
    }
    writeln!(writer)?;

    // stride = width * channels
    let row_stride = width * channels;

    // Iterate over rows
    // Note: This logic assumes your data is perfect size. Add checks if needed.
    let num_rows = data.len() / row_stride;

    for r in 0..num_rows {
        for c in 0..width {
            // Start of this pixel
            let pixel_idx = (r * row_stride) + (c * channels);

            write!(writer, "{},{}", r, c)?;

            // Write all channels for this pixel on one line
            for ch in 0..channels {
                write!(writer, ",{}", data[pixel_idx + ch])?;
            }
            writeln!(writer)?;
        }
    }

    writer.flush()?;
    Ok(())
}
