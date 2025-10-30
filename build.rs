use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Specify your shader directory relative to the project root
    let shader_dir = manifest_dir.join("shaders");

    // Process all .metal files
    let metal_files: Vec<_> = std::fs::read_dir(&shader_dir)
        .expect("Failed to read shader directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "metal" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    // Compile and embed each shader
    for metal_file in metal_files {
        compile_metal_shader(&metal_file, &out_dir)?;
    }

    // Tell cargo to rerun if shaders change
    println!("cargo:rerun-if-changed=shaders/");

    Ok(())
}

fn compile_metal_shader(
    metal_file: &Path,
    out_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_stem = metal_file.file_stem().unwrap().to_str().unwrap();
    let air_path = out_dir.join(format!("{file_stem}.air"));
    let metallib_path = out_dir.join(format!("{file_stem}.metallib"));

    // Compile .metal to .air
    let status = Command::new("xcrun")
        .args([
            "metal",
            // "-Ofast",
            "-frecord-sources",
            "-gline-tables-only",
            "-c",
            metal_file.to_str().unwrap(),
            "-o",
            air_path.to_str().unwrap(),
        ])
        .status()?;

    if !status.success() {
        return Err("Failed to compile Metal shader to .air".into());
    }

    // Compile .air to .metallib
    let status = Command::new("xcrun")
        .args([
            "metallib",
            air_path.to_str().unwrap(),
            "-o",
            metallib_path.to_str().unwrap(),
        ])
        .status()?;

    if !status.success() {
        return Err("Failed to create metallib".into());
    }

    Ok(())
}
