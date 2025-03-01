
using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering.Universal;

public class PropertiesCollector : MonoBehaviour
{
    private static readonly int BlackHolePos = Shader.PropertyToID("_BlackHolePos");
    private static readonly int BlackHoleUp = Shader.PropertyToID("_BlackHoleUp");
    private static readonly int BlackHoleForward = Shader.PropertyToID("_BlackHoleForward");
    private static readonly int Resolution = Shader.PropertyToID("_Resolution");
    public Material material;
    public GameObject obj;
    public new Camera camera;

    private TemporalAA.Settings _taaSettings;
    private Vector2 _screenResolution;
    private void Awake()
    {
        _taaSettings = camera.GetUniversalAdditionalCameraData().taaSettings;
        _taaSettings.baseBlendFactor = 1;
        
        _screenResolution = new float2(Screen.width, Screen.height);
        material.SetVector(Resolution, _screenResolution);
    }

    private void Update()
    {
        //获取obj本地坐标y轴在世界空间下的方向
        material.SetVector(BlackHolePos, obj.transform.position);
        material.SetVector(BlackHoleUp,  obj.transform.up);
        material.SetVector(BlackHoleForward,  obj.transform.forward);
        
        //检测分辨率是否发生变化
        if (!Mathf.Approximately(_screenResolution.x, Screen.width) || !Mathf.Approximately(_screenResolution.y, Screen.height))
        {
            _screenResolution = new float2(Screen.width, Screen.height);
            material.SetVector(Resolution, _screenResolution);
        }
    }
}
