struct Bytecode {
    data: array<u32, 16>,
}

struct FragmentOutput {
    @location(0) FragColor: vec4<f32>,
}

const OP_NOP: u32 = 0u;
const OP_ADD: u32 = 1u;
const OP_SUB: u32 = 2u;
const OP_MUL: u32 = 3u;
const OP_PUSH_I32: u32 = 4u;
const OP_PUSH_F64: u32 = 5u;
const OP_SIGMOID: u32 = 6u;
const OP_SQRT: u32 = 7u;
const OP_PARAM: u32 = 8u;

@group(0) @binding(0)
var<storage> bytecode: Bytecode;

@group(0) @binding(1)
var<uniform> bytecode_length: i32;

@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn execute_bytecode() -> f32 {
    var color: vec4<f32> = vec4(0f);
    var pc: i32 = 0i;
    var stack: array<f32, 64>;
    var sp: i32 = 0i;
    // var op: i32;
    // var value: i32;
    // var id: i32;
    var is_inside: f32;

    while (pc < bytecode_length) {
        var op: u32 = bytecode.data[pc];
        pc++;

        switch (op) {
            case OP_ADD: {
                sp--;
                stack[sp - 1] = stack[sp - 1] + stack[sp];
                break;
            }
            case OP_SUB: {
                sp--;
                stack[sp - 1] = stack[sp - 1] - stack[sp];
                break;
            }
            case OP_MUL: {
                sp--;
                stack[sp - 1] = stack[sp - 1] - stack[sp];
                break;
            }
            case OP_PUSH_I32: {
                var value: f32 = f32(bytecode.data[pc]);
                pc++;
                stack[sp] = value;
                sp++;
                break;
            }
            case OP_SQRT: {
                stack[sp] = sqrt(stack[sp]);
                break;
            }
            case OP_SIGMOID: {
                stack[sp] = 1.0 / (1.0 + exp(-stack[sp]));
                break;
            }
            default: {
            }
        }
    }
    return stack[sp-1];
}

@compute
@workgroup_size(16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    output[global_id.x] = execute_bytecode();
}
