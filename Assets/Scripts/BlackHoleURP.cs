using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class BlackHoleUrp : ScriptableRendererFeature
{
    class CustomRenderPass : ScriptableRenderPass
    {
        private readonly Material _material;
        private RTHandle _source;
        private RTHandle _tempTexture;
        private readonly ProfilingSampler _profilingSampler;

        public CustomRenderPass(Material material)
        {
            _material = material;
            _profilingSampler = new ProfilingSampler(nameof(CustomRenderPass));
            renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;
        }

        public void Setup(RTHandle sourceRT)
        {
            _source = sourceRT;
        }

        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            // 根据相机描述符动态分配RTHandle
            var desc = renderingData.cameraData.cameraTargetDescriptor;
            desc.depthBufferBits = 0;
            RenderingUtils.ReAllocateIfNeeded(
                ref _tempTexture, 
                desc, 
                FilterMode.Bilinear, 
                TextureWrapMode.Clamp, 
                name: "_BlackHoleTempTexture"
            );
            if (_source != null)
            {
                ConfigureTarget(_source);
            }
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
#if UNITY_EDITOR
            if (!EditorApplication.isPlaying)
            {
                return;
            }
#endif
            
            if (!_material || _tempTexture == null) return;

            var cmd = CommandBufferPool.Get();
            using (new ProfilingScope(cmd, _profilingSampler))
            {
                if (_source.rt)
                {
                    cmd.Blit(_source, _tempTexture, _material);
                    cmd.Blit(_tempTexture, _source);
                }
            }
    
            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            CommandBufferPool.Release(cmd);
        }

        public override void OnCameraCleanup(CommandBuffer cmd)
        {
            if (_tempTexture != null)
            {
                // 释放RTHandle资源
                _tempTexture.Release();
                _tempTexture = null;
            }
        }
    }

    [System.Serializable]
    public class Settings
    {
        public Material material;
    }

    public Settings settings = new();
    private CustomRenderPass _renderPass;

    public override void Create()
    {
        _renderPass = new CustomRenderPass(settings.material);
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        if (settings.material)
        {
            renderer.EnqueuePass(_renderPass);
            _renderPass.ConfigureInput(ScriptableRenderPassInput.Color);
        }
    }

    public override void SetupRenderPasses(ScriptableRenderer renderer, in RenderingData renderingData)
    {
        if (renderingData.cameraData.cameraType == CameraType.Game)
        {
            _renderPass.Setup(renderer.cameraColorTargetHandle);
        }
    }
}