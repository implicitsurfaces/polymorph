use serde_json::Value;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=opcodes.json");

    let json_file = File::open("opcodes.json").expect("Failed to open opcodes.json");
    let json: Value = serde_json::from_reader(json_file).expect("Failed to parse JSON");

    let ops = json.as_array().expect("JSON should be an array");

    let mut out_code = String::new();

    for (index, op) in ops.iter().enumerate() {
        let op_name = op.as_str().expect("Each item should be a string");
        out_code.push_str(&format!("pub const {}: u32 = {};\n", op_name, index));
    }

    // Write the generated code to opcodes.rs.
    // Then it can be included like so:
    //     include!(concat!(env!("OUT_DIR"), "/opcodes.rs"));
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("opcodes.rs");
    let mut file = File::create(dest_path).expect("Failed to create output file");
    file.write_all(out_code.as_bytes())
        .expect("Failed to write to file");
}
