use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders/conv3x3.comp.glsl");

    let out_dir = env::var("OUT_DIR").unwrap();

    // Invoke the CLI tool
    let status = Command::new("glslangValidator")
        .args(&[
            "-V", // Generate SPIR-V
            "shaders/conv3x3.comp.glsl",
            "-o",
            &format!("{}/conv3x3.comp.spv", out_dir),
        ])
        .status()
        .expect("Failed to run glslangValidator");

    if !status.success() {
        panic!("Shader compilation failed!");
    }
}
