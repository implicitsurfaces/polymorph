use serde_json::Value;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=opcodes.json");
    println!("cargo:rerun-if-changed=shader-in.wgsl");

    // Read opcodes (a list of strings) from opcodes.json.
    let json_file = File::open("opcodes.json").expect("Failed to open opcodes.json");
    let json: Value = serde_json::from_reader(json_file).expect("Failed to parse JSON");
    let ops = json.as_array().expect("JSON should be an array");

    // Generate Rust and WGSL contants for each opcode.
    let mut rs_out = String::new();
    let mut wgsl_out = String::new();

    for (index, op) in ops.iter().enumerate() {
        let op_name = op.as_str().expect("Each item should be a string");
        rs_out.push_str(&format!("pub const {}: u32 = {};\n", op_name, index));
        wgsl_out.push_str(&format!("\nconst {}: u32 = {}u;", op_name, index));
    }

    // Write the generated Rust code to the build directory.
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let rs_dest_path = Path::new(&out_dir).join("opcodes.rs");
    let mut rs_file = File::create(rs_dest_path).expect("Failed to create output file");
    rs_file
        .write_all(rs_out.as_bytes())
        .expect("Failed to write to Rust file");

    // Write the modified shader.wgsl to the build directory.
    let mut shader_content =
        std::fs::read_to_string("src/shader-in.wgsl").expect("Failed to read shader.wgsl");
    shader_content = shader_content.replace("$generated_prelude_goes_here", &wgsl_out);
    let wgsl_dest_path = Path::new(&out_dir).join("shader.wgsl");
    std::fs::write(wgsl_dest_path, shader_content).expect("Failed to write to shader.wgsl");
}
