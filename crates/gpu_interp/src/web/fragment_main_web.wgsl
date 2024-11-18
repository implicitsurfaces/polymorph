fn viridis(t: f32) -> vec4<f32> {
    // Coefficients from https://www.shadertoy.com/view/WlfXRN
    let c0 = vec4<f32>(0.2777273272234177, 0.005407344544966578, 0.3340998053353061, 1.0);
    let c1 = vec4<f32>(0.1050930431085774, 1.404613529898575, 1.384590162594685, 0.0);
    let c2 = vec4<f32>(-0.3308618287255563, 0.214847559468213, 0.09509516302823659, 0.0);
    let c3 = vec4<f32>(-4.634230498983486, -5.799100973351585, -19.33244095627987, 0.0);
    let c4 = vec4<f32>(6.228269936347081, 14.17993336680509, 56.69055260068105, 0.0);
    let c5 = vec4<f32>(-4.776830223517424, -13.21772673655391, -55.13031880482764, 0.0);
    let c6 = vec4<f32>(1.225428807585486, 4.1773687843226, 16.438632580884, 0.0);

    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let t6 = t5 * t;

    return c0 + c1 * t + c2 * t2 + c3 * t3 + c4 * t4 + c5 * t5 + c6 * t6;
}


fn diverging_purple_orange(t: f32, min_t: f32, max_t: f32) -> vec4<f32> {
    // t in (-1, 1)
    if (t <= 0.0) {
        let purple_t = t / min_t;
        return vec4<f32>(
            0.404 + (1.0 - 0.404) * (1 - purple_t),  // R: interpolate from purple to white
            0.039 + (1.0 - 0.039) * (1 - purple_t),  // G: interpolate from purple to white
            0.561 + (1.0 - 0.561) * (1 - purple_t),  // B: interpolate from purple to white
            1.0                                      // A
        );
    } else {
        let orange_t = t / max_t;
        return vec4<f32>(
            0.880 + (1.0 - 0.880) * (1 - orange_t),  // R: interpolate from orange to white
            0.510 + (1.0 - 0.510) * (1 - orange_t),  // G: interpolate from orange to white
            0.016 + (1.0 - 0.016) * (1 - orange_t),  // B: interpolate from orange to white
            1.0                                      // A
        );
    }
}




@fragment
fn fragment_main_web(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
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
    let pixel_group = output[y * row_len + buf_x];

    var pixel_value = pixel_group[x % 4u];

    // max visible distance is +/- viewport length (it's square)
    let pmax = f32(viewport[0]) / 2.0;

    return diverging_purple_orange(pixel_value / pmax, -0.3, 1.0);
}
