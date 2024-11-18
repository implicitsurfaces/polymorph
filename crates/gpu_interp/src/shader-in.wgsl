// We do some very basic template substitution here when the shader is
// loaded, to allow constants to be shared between Rust and WGSL.
{ shared_constants }

struct Bytecode {
    data: array<u32, BYTECODE_ARRAY_LEN>,
}

struct FragmentOutput {
    @location(0) FragColor: vec4<f32>,
}

struct Projection {
    scale: vec2<f32>,
    translation: vec2<f32>,
}

@group(0) @binding(0) var<storage> bytecode: Bytecode;

@group(0) @binding(1) var<uniform> bc_offsets: array<vec4<u32>, MAX_TILE_COUNT_DIV_4>;

@group(0) @binding(2) var<uniform> bc_ends: array<vec4<u32>, MAX_TILE_COUNT_DIV_4>;

@group(0) @binding(3) var<storage, read_write> output: array<vec4<f32>>;

@group(0) @binding(4) var<uniform> viewport: vec2<u32>;

@group(0) @binding(5) var<uniform> step_count: u32;

@group(0) @binding(6) var<uniform> projection: Projection;

fn execute_bytecode(xs: vec4<f32>, y: f32, tile_idx: u32) -> vec4<f32> {
    var reg: array<vec4<f32>, REG_COUNT>;

    // Uniforms need 16-byte alignment, so we use a vec4<u32>.
    var pc = bc_offsets[tile_idx / 4u][tile_idx % 4u];
    let pc_max = bc_ends[tile_idx / 4u][tile_idx % 4u];

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
            case 1u /* Input */: {
              let out_reg = lo[1];
              let i = bitcast<u32>(hi);
              if (i == 0) {
                reg[out_reg] = (xs * projection.scale.x) - projection.translation.x;
              } else if (i == 1) {
                reg[out_reg] = vec4<f32>(y * projection.scale.y) - projection.translation.y;
              }
            }
            case 0u /* Output */: {
              let src_reg = lo[1];
              let i = bitcast<u32>(hi);
              if (i == 0) {
                return reg[src_reg];
              }
            }
            case 2u /* CopyReg */: { reg[lo[1]] = reg[lo[2]]; }
            case 4u /* NegReg */: { reg[lo[1]] = -reg[lo[2]]; }
            case 7u /* SqrtReg */: { reg[lo[1]] = sqrt(reg[lo[2]]); }
            case 8u /* SquareReg */: {
              let val = reg[lo[2]];
              reg[lo[1]] = val * val;
            }
            case 21u /* AddRegImm */: { reg[lo[1]] = reg[lo[2]] + hi; }
            case 22u /* MulRegImm */: { reg[lo[1]] = reg[lo[2]] * hi; }
            case 29u /* SubImmReg */: { reg[lo[1]] = hi - reg[lo[2]]; }
            case 24u /* SubRegImm */: { reg[lo[1]] = reg[lo[2]] - hi; }
            case 33u /* MinRegImm */: { reg[lo[1]] = min(reg[lo[2]], vec4<f32>(hi)); }
            case 34u /* MaxRegImm */: { reg[lo[1]] = max(reg[lo[2]], vec4<f32>(hi)); }
            case 37u /* AddRegReg */: { reg[lo[1]] = reg[lo[2]] + reg[lo[3]]; }
            case 38u /* MulRegReg */: { reg[lo[1]] = reg[lo[2]] * reg[lo[3]]; }
            case 40u /* SubRegReg */: { reg[lo[1]] = reg[lo[2]] - reg[lo[3]]; }
            case 44u /* MinRegReg */: { reg[lo[1]] = min(reg[lo[2]], reg[lo[3]]); }
            case 45u /* MaxRegReg */: { reg[lo[1]] = max(reg[lo[2]], reg[lo[3]]); }
            default: {
              return vec4<f32>(669.0, f32(pc), f32(lo[0]), f32(lo[0]));
            }
          }
    }
    return vec4<f32>(0.0);
}

@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each shader invocation processes 4 horizontal pixels, and the output
    // is a vec4<f32> representing four pixels.
    let row_len = viewport[0] / 4u;
    let tile_row_len = viewport[0] / TILE_SIZE_X;

    let tile_x = global_id.x * 4u / TILE_SIZE_X; // tile size is in pixels
    let tile_y = global_id.y / TILE_SIZE_Y;

    let xs = vec4<f32>(f32(global_id.x) * 4.0) + vec4<f32>(0.0, 1.0, 2.0, 3.0);
    let out_idx = global_id.y * row_len + global_id.x;
    let tile_idx = tile_y * tile_row_len + tile_x;
    output[out_idx] = execute_bytecode(xs, f32(global_id.y), tile_idx);
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

    // nothing for us to do outside of the viewport
    if (x >= viewport[0] || y >= viewport[1]) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // Each shader invocation processes 4 horizontal pixels, and the output
    // is a vec4<f32> representing four pixels.
    let row_len = viewport[0] / 4u;
    let buf_x = x / 4u;
    let offset = x % 4u;
    let index = y * row_len + buf_x;

    let pixel_group = output[index];
    let is_inside = f32(pixel_group[offset] < 0.0);

    return vec4<f32>(is_inside, is_inside, is_inside, 1.0);
}
