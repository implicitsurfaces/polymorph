// We do some very basic template substitution here when the shader is
// loaded, to allow constants to be shared between Rust and WGSL.
{ shared_constants }

struct Bytecode {
    data: array<u32, BYTECODE_ARRAY_LEN>,
}

struct FragmentOutput {
    @location(0) FragColor: vec4<f32>,
}

@group(0) @binding(0)
var<storage> bytecode: Bytecode;

@group(0) @binding(1)
var<uniform> pc_max: i32;

@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

@group(0) @binding(3) var<uniform> dims: vec2<u32>;

@group(0) @binding(4) var<uniform> step_count: u32;

fn execute_bytecode(xs: vec4<f32>, y: u32) -> vec4<f32> {
    var pc: i32 = 0;
    var reg: array<vec4<f32>, REG_COUNT>;
    var mem: array<vec4<f32>, MEM_SIZE>;
    while (pc < pc_max) {
        /*
          Memory layout notes:
          - On the Rust side, the bytecode is seralized into a Vec<u8>. When
            we say "byte 0" or "first 4 bytes" we mean `bc_vec[0]` and
            `bc_vec[0..4]`, respectively.
          - Ops are 8 bytes long, with the first byte (i.e. `lo[0]`, below)
            being the opcode.
          - u8 arguments are generally packed into the 3 bytes following the
            opcode.
          - 32-bit arguments (u32 and f32) are stored in little-endian order
            in bytes 4-7.
          - Thus, `lo` is a 4xU8 while `hi` is an f32.
        */
        let lo: vec4<u32> = unpack4xU8(bytecode.data[pc]);
        pc++;
        let hi: f32 = bitcast<f32>(bytecode.data[pc]);
        pc++;

        switch (lo[0]) {
            case 0u /* Input */: {
              let out_reg = lo[1];
              let i = bitcast<u32>(hi);
              if (i == 0) {
                reg[out_reg] = (xs - 750.0) / 500.0;
              } else if (i == 1) {
                reg[out_reg] = (vec4<f32>(y) - 750.0) / 500.0;
              }
            }
            case 1u /* Output */: {
              let src_reg = lo[1];
              let i = bitcast<u32>(hi);
              if (i == 0) {
                return reg[src_reg];
              }
            }
            case 2u /* NegReg */: { reg[lo[1]] = -reg[lo[2]]; }
            case 5u /* SqrtReg */: { reg[lo[1]] = sqrt(reg[lo[2]]); }
            case 6u /* SquareReg */: {
              let val = reg[lo[2]];
              reg[lo[1]] = val * val;
            }
            case 20u /* AddRegImm */: { reg[lo[1]] = reg[lo[2]] + hi; }
            case 21u /* MulRegImm */: { reg[lo[1]] = reg[lo[2]] * hi; }
            case 24u /* SubImmReg */: { reg[lo[1]] = hi - reg[lo[2]]; }
            case 25u /* SubRegImm */: { reg[lo[1]] = reg[lo[2]] - hi; }
            case 32u /* MinRegImm */: { reg[lo[1]] = min(reg[lo[2]], vec4<f32>(hi)); }
            case 33u /* MaxRegImm */: { reg[lo[1]] = max(reg[lo[2]], vec4<f32>(hi)); }
            case 38u /* AddRegReg */: { reg[lo[1]] = reg[lo[2]] + reg[lo[3]]; }
            case 39u /* MulRegReg */: { reg[lo[1]] = reg[lo[2]] * reg[lo[3]]; }
            case 41u /* SubRegReg */: { reg[lo[1]] = reg[lo[2]] - reg[lo[3]]; }
            case 42u /* MinRegReg */: { reg[lo[1]] = min(reg[lo[2]], reg[lo[3]]); }
            case 43u /* MaxRegReg */: { reg[lo[1]] = max(reg[lo[2]], reg[lo[3]]); }
            case 48u /* Load */: { reg[lo[1]] = mem[bitcast<u32>(hi)]; }
            case 49u /* Store */: { mem[bitcast<u32>(hi)] = reg[lo[1]]; }
            default: {
              return vec4<f32>(1.234567);
            }
          }
    }
    return vec4<f32>(99.0);
}

@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each shader invocation processes 4 horizontal pixels, and the output
    // is a vec4<f32> representing four pixels.
    let quarter_width = dims.x / 4u;

    let index = global_id.y * quarter_width + global_id.x;
    let xs = vec4<f32>(f32(global_id.x) * 4.0) + vec4<f32>(0.0, 1.0, 2.0, 3.0);
    output[index] = execute_bytecode(xs, global_id.y);
}

@vertex
fn vertex_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Create vertices for two triangles that form a quad
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),  // Triangle 1
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),  // Triangle 2
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );

    return vec4<f32>(pos[in_vertex_index], 0.0, 1.0);
}

@fragment
fn fragment_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(pos.x);
    let y = u32(pos.y);

    // Each shader invocation processes 4 horizontal pixels, and the output
    // is a vec4<f32> representing four pixels.
    let row_len = dims.x / 4u;
    let buf_x = x / 4u;
    let offset = x % 4u;
    let index = y * row_len + buf_x;

    let pixel_group = output[index];
    let is_inside = f32(pixel_group[offset] < 0.0);

    return vec4<f32>(is_inside, is_inside, is_inside, 1.0);
}
