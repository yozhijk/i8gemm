use std::env;
use std::fs;
use std::process::Command;

fn main() {
    let shader_dir = "shaders";
    let out_dir = env::var("OUT_DIR").unwrap();

    println!("cargo:rerun-if-changed={}", shader_dir);

    let entries =
        fs::read_dir(shader_dir).expect("Could not read 'shaders' directory. Does it exist?");

    for entry in entries {
        let entry = entry.expect("Error reading directory entry");
        let path = entry.path();

        if path.is_file() && path.extension().map_or(false, |ext| ext == "glsl") {
            let file_name = path.file_name().unwrap().to_str().unwrap();

            println!("cargo:rerun-if-changed={}", path.display());

            let file_stem = path.file_stem().unwrap().to_str().unwrap();
            let out_path = format!("{}/{}.spv", out_dir, file_stem);

            let status = Command::new("glslangValidator")
                .args(&[
                    "-V",
                    "--target-env",
                    "vulkan1.4",
                    path.to_str().unwrap(),
                    "-o",
                    &out_path,
                ])
                .status()
                .expect("Failed to execute glslangValidator. Is it in your PATH?");

            if !status.success() {
                // Panic ensures the build fails so you see the shader error
                panic!("Shader compilation failed for: {}", file_name);
            }
        }
    }
}
