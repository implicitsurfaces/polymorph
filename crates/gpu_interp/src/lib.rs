// TODO: actual shared code here. Just adding this constant to test notebook integration.
pub const HELLO: &'static str = "WORLD";

pub async fn create_device(
    instance: &wgpu::Instance,
    options: &wgpu::RequestAdapterOptions<'_, '_>,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance.request_adapter(options).await.unwrap();

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::SHADER_INT64
                    | wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .unwrap();

    (adapter, device, queue)
}
