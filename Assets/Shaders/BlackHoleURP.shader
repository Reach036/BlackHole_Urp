// Shader部分 URP版（RaymarchingURP.shader）
Shader "Universal Render Pipeline/Effects/BlackHoleURP"
{
    Properties
    {
        _MainTex ("Main Texture", 2D) = "white" {}
        _BlackHolePos ("黑洞位置", Vector) = (0, 0, -10)
        _BlackHoleUp ("吸积盘法向", Vector) = (0, 1, 0)
        _BlackHoleForward ("黑洞朝向", Vector) = (0, 1, 0)
        _Resolution ("分辨率", Vector) = (1920, 1080, 1, 1)
        _Radius ("吸积盘半径", Float) = 12
        _Mass ("黑洞质量（单位为一个太阳质量）", Float) = 14900000
        _Bloom ("Bloom", float) = 12
        _Speed ("黑洞转速", int) = 30
        _Iterations ("迭代次数", int) = 150
        _DmdtMulti ("吸积率", Range(0.00000001, 0.1)) = 0.000002
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
        }

        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

        // 预计算常量
        #define SPEED_OF_LIGHT 2.99792458e8
        #define LIGHT_YEAR 9.4607e15
        #define GRAVITY_CONSTANT 6.67430e-11
        #define SOLAR_MASS 1.98847e30
        #define SIGMA 5.670373e-8
        #define SHIFT_MAX 1.25
        #define RANDOM_SEED 43758.5453

        struct Attributes
        {
            float4 positionHCS : POSITION;
            float2 uv : TEXCOORD0;
        };

        struct Varyings
        {
            float4 positionCS : SV_POSITION;
            float2 uv : TEXCOORD0;
            float3 viewDir : TEXCOORD1;
        };

        TEXTURE2D(_MainTex);
        SAMPLER(sampler_MainTex);

        CBUFFER_START(UnityPerMaterial)
            float3 _BlackHolePos;
            float3 _BlackHoleUp;
            float3 _BlackHoleForward;
            float _Radius;
            float _Mass;
            float2 _Resolution;
            float _DmdtMulti;
            int _Bloom;
            int _Iterations;
            int _Speed;
        CBUFFER_END

        //使用宏定义替代重复计算
        #define GET_DISK_TEMPERATURE(posR, diskA) pow(diskA * Rs * Rs * Rs / (posR * posR * posR) * max(1.0 - sqrt(InterRadius / posR), 0.000001), 0.25)
        #define GET_REDSHIFT(posR, cameraPos) sqrt(max(1.0 - Rs / posR, 0.000001)) / sqrt(max(1.0 - Rs / length(cameraPos), 0.000001))

        inline float RandomStep(float2 xy, float seed)
        {
            return frac(sin(dot(xy + frac(11.4514 * sin(seed)), float2(12.9898, 78.233))) * RANDOM_SEED);
        }

        inline float3 UvToDir(float2 uv)
        {
            float4 clipSpacePos = float4(uv * 2.0 - 1.0, -1.0, 1.0);
            float4 viewSpacePos = mul(unity_CameraInvProjection, clipSpacePos);
            return normalize(viewSpacePos.xyz / viewSpacePos.w);
        }

        float2 DirToUV(float3 dir)
        {
            return float2(0.5 - 0.5 * dir.x / dir.z, 0.5 - 0.5 * dir.y / dir.z * _Resolution.x / _Resolution.y);
        }

        float CubicInterpolate(float x)
        {
            return 3.0 * x * x - 2.0 * x * x * x;
        }

        float PerlinNoise(float3 Position)
        {
            float3 PosInt = floor(Position);
            float3 PosFloat = frac(Position);
            float3 noiseParams = float3(12.9898, 78.233, 213.765);

            float v000 = 2.0 * frac(sin(dot(float3(PosInt.x, PosInt.y, PosInt.z), noiseParams)) * RANDOM_SEED) - 1.0;
            float v100 = 2.0 * frac(sin(dot(float3(PosInt.x + 1.0, PosInt.y, PosInt.z), noiseParams)) * RANDOM_SEED) - 1.0;
            float v010 = 2.0 * frac(sin(dot(float3(PosInt.x, PosInt.y + 1.0, PosInt.z), noiseParams)) * RANDOM_SEED) - 1.0;
            float v110 = 2.0 * frac(sin(dot(float3(PosInt.x + 1.0, PosInt.y + 1.0, PosInt.z), noiseParams)) * RANDOM_SEED) - 1.0;
            float v001 = 2.0 * frac(sin(dot(float3(PosInt.x, PosInt.y, PosInt.z + 1.0), noiseParams)) * RANDOM_SEED) - 1.0;
            float v101 = 2.0 * frac(sin(dot(float3(PosInt.x + 1.0, PosInt.y, PosInt.z + 1.0), noiseParams)) * RANDOM_SEED) - 1.0;
            float v011 = 2.0 * frac(sin(dot(float3(PosInt.x, PosInt.y + 1.0, PosInt.z + 1.0), noiseParams)) * RANDOM_SEED) - 1.0;
            float v111 = 2.0 * frac(sin(dot(float3(PosInt.x + 1.0, PosInt.y + 1.0, PosInt.z + 1.0), noiseParams)) * RANDOM_SEED) - 1.0;

            float v00 = v001 * CubicInterpolate(PosFloat.z) + v000 * CubicInterpolate(1.0 - PosFloat.z);
            float v10 = v101 * CubicInterpolate(PosFloat.z) + v100 * CubicInterpolate(1.0 - PosFloat.z);
            float v01 = v011 * CubicInterpolate(PosFloat.z) + v010 * CubicInterpolate(1.0 - PosFloat.z);
            float v11 = v111 * CubicInterpolate(PosFloat.z) + v110 * CubicInterpolate(1.0 - PosFloat.z);
            float v0 = v01 * CubicInterpolate(PosFloat.y) + v00 * CubicInterpolate(1.0 - PosFloat.y);
            float v1 = v11 * CubicInterpolate(PosFloat.y) + v10 * CubicInterpolate(1.0 - PosFloat.y);
            return v1 * CubicInterpolate(PosFloat.x) + v0 * CubicInterpolate(1.0 - PosFloat.x);
        }

        float SoftSaturate(float x)
        {
            return 1.0 - 1.0 / (max(x, 0.0) + 1.0);
        }

        #define SAMPLE_NOISE(pos, level) PerlinNoise(float3(pos) * pow(3.0, float(level)))

        float GenerateAccretionDiskNoise(float3 Position, int NoiseStartLevel, int NoiseEndLevel, float ContrastLevel)
        {
            float NoiseAccumulator = 10.0;
            [unroll]
            for (int Level = NoiseStartLevel; Level < NoiseEndLevel; ++Level)
            {
                NoiseAccumulator *= 1.0 + 0.1 * SAMPLE_NOISE(Position, Level);
            }
            return log(1.0 + pow(0.1 * NoiseAccumulator, ContrastLevel));
        }

        float Vec2ToTheta(float2 v1, float2 v2)
        {
            float dotProduct = dot(v1, v2);
            float crossProduct = v1.x * v2.y - v1.y * v2.x;
            float angle = asin(0.999999 * crossProduct / (length(v1) * length(v2)));
            return dotProduct > 0.0 ? angle : crossProduct < 0.0 ? PI - angle : -PI - angle;
        }

        float3 KelvinToRgb(float Kelvin)
        {
            if (Kelvin < 400.01)
            {
                return float3(0, 0, 0);
            }

            float Teff = (Kelvin - 6500.0) / (6500.0 * Kelvin * 2.2);
            float3 RgbColor = float3(0, 0, 0);;

            RgbColor.r = exp(2.05539304e4 * Teff);
            RgbColor.g = exp(2.63463675e4 * Teff);
            RgbColor.b = exp(3.30145739e4 * Teff);

            float BrightnessScale = 1.0 / max(max(RgbColor.r, RgbColor.g), RgbColor.b);

            if (Kelvin < 1000.0)
            {
                BrightnessScale *= (Kelvin - 400.0) / 600.0;
            }

            RgbColor *= BrightnessScale;
            return RgbColor;
        }

        float GetKeplerianAngularVelocity(float Radius, float Rs)
        {
            return sqrt(SPEED_OF_LIGHT / LIGHT_YEAR * SPEED_OF_LIGHT * Rs / LIGHT_YEAR / ((2.0 * Radius - 3.0 * Rs) * Radius * Radius));
        }

        // 计算相机系下旋转和平移
        float3 GetCameraTransform(float4 Position)
        {
            float4 camSpacePos = mul(unity_WorldToCamera, Position);
            camSpacePos.z *= -1;
            return camSpacePos.xyz;
        }

        float3 GetCameraRot(float4 targetNormal)
        {
            float3 camSpaceNormal = mul((float3x3)unity_WorldToCamera, targetNormal.xyz);
            camSpaceNormal.z *= -1;
            return normalize(camSpaceNormal);
        }

        //相机系下的世界坐标转换到黑洞坐标系
        float3 WorldToBlackHoleSpace(float4 Position, float3 BlackHolePos, float3 DiskForward, float3 DiskNormal)
        {
            // 3. 构建局部坐标系基向量
            float3 localY = normalize(DiskNormal); // Y轴：法线方向
            float3 localZ = normalize(DiskForward); // Z轴：
            float3 localX = normalize(cross(localY, localZ)); // X轴：右手法则确定

            // 4. 构造旋转矩阵（基向量作为列向量，直接用于坐标变换）
            float3x3 rotation = float3x3(localX, localY, localZ);

            // 5. 应用变换：先平移到黑洞中心，再旋转到局部坐标系
            float3 pos = Position.xyz - BlackHolePos;
            return mul(rotation, pos);
        }

        //相机系下的矢量转换到黑洞坐标系
        float3 ApplyBlackHoleRotation(float4 Direction, float3 DiskForward, float3 DiskNormal)
        {
            // ===== 构建局部坐标系（右手系） =====
            float3 localY = normalize(DiskNormal); // Y轴：法线方向
            float3 localZ = normalize(DiskForward); // Z轴
            float3 localX = normalize(cross(localY, localZ)); // X轴：右手法则

            // ===== 构造旋转矩阵（Unity行优先矩阵） =====
            float3x3 rotation = float3x3(
                localX.x, localY.x, localZ.x, // 第一行：X轴
                localX.y, localY.y, localZ.y, // 第二行：Y轴
                localX.z, localY.z, localZ.z // 第三行：Z轴
            );

            return mul(Direction, rotation); // Unity行优先矩阵左乘
        }

        //吸积盘形状
        float Shape(float x, float Alpha, float Beta)
        {
            float k = pow(Alpha + Beta, Alpha + Beta) / (pow(Alpha, Alpha) * pow(Beta, Beta));
            return k * pow(x, Alpha) * pow(1.0 - x, Beta);
        }

        //吸积盘着色
        float4 DiskColor(float4 BaseColor, float TimeRate, float StepLength, float3 RayPos, float3 RayDir, float3 BlackHolePos, float3 DiskForward, float3 DiskNormal,
                         float Rs, float InterRadius, float OuterRadius, float DiskTemperatureArgument, float QuadraticedPeakTemperature, float ShiftMax)
        {
            float3 CameraPos = WorldToBlackHoleSpace(float4(0, 0, 0, 1), BlackHolePos, DiskForward, DiskNormal);
            float3 PosOnDisk = WorldToBlackHoleSpace(float4(RayPos, 1), BlackHolePos, DiskForward, DiskNormal);
            float3 DirOnDisk = ApplyBlackHoleRotation(float4(RayDir, 0), DiskForward, DiskNormal);

            float PosR = length(PosOnDisk.xz);
            float PosY = PosOnDisk.y;

            float4 Color = float4(0.0, 0.0, 0.0, 0.0);
            if (abs(PosY) < 0.5 * Rs && PosR < OuterRadius && PosR > InterRadius)
            {
                float EffectiveRadius = 1.0 - (PosR - InterRadius) / (OuterRadius - InterRadius) * 0.5;
                if (OuterRadius - InterRadius > 9.0 * Rs)
                {
                    if (PosR < 5.0 * Rs + InterRadius)
                    {
                        EffectiveRadius = 1.0 - (PosR - InterRadius) / (9.0 * Rs) * 0.5;
                    }
                    else
                    {
                        EffectiveRadius = 1.0 - (0.5 / 0.9 * 0.5 + ((PosR - InterRadius) / (OuterRadius - InterRadius) -
                            5.0 * Rs / (OuterRadius - InterRadius)) / (1.0 - 5.0 * Rs / (OuterRadius - InterRadius)) * 0.5);
                    }
                }

                if (abs(PosY) < 0.5 * Rs * Shape(EffectiveRadius, 4.0, 0.9) || PosY < 0.5 * Rs * (1.0 - 5.0 * pow(2.0 * (1.0 - EffectiveRadius), 2.0)))
                {
                    float AngularVelocity = GetKeplerianAngularVelocity(PosR, Rs);
                    float HalfPiTimeInside = PI / GetKeplerianAngularVelocity(3.0 * Rs, Rs);
                    float EffectiveTime0 = frac(_Time.y * TimeRate / HalfPiTimeInside) * HalfPiTimeInside + 0.0 * HalfPiTimeInside;
                    float EffectiveTime1 = frac(_Time.y * TimeRate / HalfPiTimeInside + 0.5) * HalfPiTimeInside + 1.0 * HalfPiTimeInside;
                    float PhaseTimeIndex0 = trunc(_Time.y * TimeRate / HalfPiTimeInside);
                    float PhaseTimeIndex1 = trunc(_Time.y * TimeRate / HalfPiTimeInside + 0.5);
                    float Phase0 = 2.0 * PI * frac(RANDOM_SEED * sin(PhaseTimeIndex0));
                    float Phase1 = 2.0 * PI * frac(RANDOM_SEED * sin(PhaseTimeIndex1));

                    float PosThetaWithoutTime = Vec2ToTheta(PosOnDisk.zx, float2(1.0, 0.0));
                    float PosTheta = frac((PosThetaWithoutTime + AngularVelocity * EffectiveTime0 + Phase0) / (2.0 * PI)) * 2.0 * PI;

                    // 计算盘温度
                    float DiskTemperature = pow(DiskTemperatureArgument * Rs * Rs * Rs / (PosR * PosR * PosR) * max(1.0 - sqrt(InterRadius / PosR), 0.000001), 0.25);
                    // 计算云相对速度
                    float3 CloudVelocity = 9.4607e15 / 2.99792458e8 * AngularVelocity * cross(float3(0., 1., 0.), PosOnDisk);
                    float RelativeVelocity = dot(-DirOnDisk, CloudVelocity);
                    // 计算多普勒因子
                    float Dopler = sqrt((1.0 + RelativeVelocity) / (1.0 - RelativeVelocity));
                    // 总红移量，含多普勒因子和引力红移和
                    float RedShift = Dopler * sqrt(max(1.0 - Rs / PosR, 0.000001)) / sqrt(max(1.0 - Rs / length(CameraPos), 0.000001));

                    float Thick;
                    float VerticalMixFactor;
                    float DustColor = 0.0;
                    float4 Color0 = float4(0.0, 0.0, 0.0, 0.0);
                    float4 Color1 = float4(0.0, 0.0, 0.0, 0.0);
                    float Density = Shape(EffectiveRadius, 4.0, 0.9);

                    float tempY = 0.5 * Rs * Density;
                    if (abs(PosY) < tempY)
                    {
                        Thick = tempY * (0.4 + 0.6 * SoftSaturate(GenerateAccretionDiskNoise(float3(1.5 * PosTheta, PosR / Rs, 1.0), 1, 3, 80.0))); // 盘厚
                        VerticalMixFactor = max(0.0, 1.0 - abs(PosY) / Thick);
                        Density *= 0.7 * VerticalMixFactor * Density;

                        float tempNoise = GenerateAccretionDiskNoise(float3(1.0 * PosR / Rs, 1.0 * PosY / Rs, 0.5 * PosTheta), 3, 6, 80.0);
                        Color0 = float4(tempNoise, tempNoise, tempNoise, tempNoise); // 云本体
                        Color0.xyz *= Density * 1.4 * (0.2 + 0.8 * VerticalMixFactor + (0.8 - 0.8 * VerticalMixFactor) *
                            GenerateAccretionDiskNoise(float3(PosR / Rs, 1.5 * PosTheta, PosY / Rs), 1, 3, 80.0));
                        Color0.a *= Density; // * (1.0 + VerticalMixFactor);
                    }
                    if (abs(PosY) < 0.5 * Rs * (1.0 - 5.0 * pow(2.0 * (1.0 - EffectiveRadius), 2.0)))
                    {
                        DustColor = max(1.0 - pow(PosY / (0.5 * Rs * max(1.0 - 5.0 * pow(2.0 * (1.0 - EffectiveRadius), 2.0), 0.0001)), 2.0), 0.0) * GenerateAccretionDiskNoise(
                            float3(1.5 * frac((PosThetaWithoutTime + PI / HalfPiTimeInside * EffectiveTime0 + Phase0) / (2.0 * PI)) * 2.0 * PI, PosR / Rs, PosY / Rs), 0, 6, 80.0);
                        Color0 += 0.02 * float4(DustColor, DustColor, DustColor, 0.2 * DustColor) * sqrt(1.0001 - DirOnDisk.y * DirOnDisk.y) * min(1.0, Dopler * Dopler);
                    }
                    Color0 *= 0.5 - 0.5 * cos(2.0 * PI * frac(_Time.y * TimeRate / HalfPiTimeInside)); // 用于过渡

                    PosTheta = frac((PosThetaWithoutTime + AngularVelocity * EffectiveTime1 + Phase1) / (2.0 * PI)) * 2.0 * PI; // 更新相位
                    Density = Shape(EffectiveRadius, 4.0, 0.9);

                    tempY = 0.5 * Rs * Density;
                    if (abs(PosY) < tempY)
                    {
                        Thick = tempY * (0.4 + 0.6 * SoftSaturate(GenerateAccretionDiskNoise(float3(1.5 * PosTheta, PosR / Rs, 1.0), 1, 3, 80.0)));
                        VerticalMixFactor = max(0.0, 1.0 - abs(PosY) / Thick);
                        Density *= 0.7 * VerticalMixFactor * Density;

                        float tempNoise = GenerateAccretionDiskNoise(float3(1.0 * PosR / Rs, 1.0 * PosY / Rs, 0.5 * PosTheta), 3, 6, 80.0);
                        Color1 = float4(tempNoise, tempNoise, tempNoise, tempNoise);
                        Color1.xyz *= Density * 1.4 * (0.2 + 0.8 * VerticalMixFactor + (0.8 - 0.8 * VerticalMixFactor) * GenerateAccretionDiskNoise(
                            float3(PosR / Rs, 1.5 * PosTheta, PosY / Rs), 1, 3, 80.0));
                        Color1.a *= Density; // * (1.0 + VerticalMixFactor);
                    }
                    if (abs(PosY) < 0.5 * Rs * (1.0 - 5.0 * pow(2.0 * (1.0 - EffectiveRadius), 2.0)))
                    {
                        DustColor = max(1.0 - pow(PosY / (0.5 * Rs * max(1.0 - 5.0 * pow(2.0 * (1.0 - EffectiveRadius), 2.0), 0.0001)), 2.0), 0.0) * GenerateAccretionDiskNoise(
                            float3(1.5 * frac((PosThetaWithoutTime + PI / HalfPiTimeInside * EffectiveTime1 + Phase1) / (2.0 * PI)) * 2.0 * PI, PosR / Rs, PosY / Rs), 0, 6, 80.0);
                        Color1 += 0.02 * float4(DustColor, DustColor, DustColor, 0.2 * DustColor) * sqrt(1.0001 - DirOnDisk.y * DirOnDisk.y) * min(1.0, Dopler * Dopler);
                    }
                    Color1 *= 0.5 - 0.5 * cos(2.0 * PI * frac(_Time.y * TimeRate / HalfPiTimeInside + 0.5));

                    Color = Color1 + Color0;
                    Color *= 1.0 + 20.0 * exp(-10.0 * (PosR - InterRadius) / (OuterRadius - InterRadius)); // 内侧增加密度

                    float BrightWithoutRedshift = 4.5 * DiskTemperature * DiskTemperature * DiskTemperature * DiskTemperature / QuadraticedPeakTemperature; // 原亮度
                    if (DiskTemperature > 1000.0)
                    {
                        DiskTemperature = max(1000.0, DiskTemperature * RedShift * Dopler * Dopler);
                    }

                    DiskTemperature = min(100000.0, DiskTemperature);

                    Color.xyz *= BrightWithoutRedshift * min(1.0, 1.8 * (OuterRadius - PosR) / (OuterRadius - InterRadius)) * KelvinToRgb(
                        DiskTemperature / exp((PosR - InterRadius) / (0.6 * (OuterRadius - InterRadius))));
                    Color.xyz *= min(ShiftMax, RedShift) * min(ShiftMax, Dopler);

                    RedShift = min(RedShift, ShiftMax);
                    Color.xyz *= pow(1.0 - (1.0 - min(1.0, RedShift)) * (PosR - InterRadius) / (OuterRadius - InterRadius), 9.0);
                    Color.xyz *= min(1.0, 1.0 + 0.5 * ((PosR - InterRadius) / InterRadius + InterRadius / (PosR - InterRadius)) - max(1.0, RedShift));

                    Color *= StepLength / Rs;
                }
            }
            return BaseColor + Color * (1.0 - BaseColor.a);
        }

        Varyings vert(Attributes input)
        {
            Varyings output;
            output.positionCS = TransformObjectToHClip(input.positionHCS.xyz);
            output.uv = input.uv;
            output.viewDir = float3(0, 0, -1);
            return output;
        }

        half4 frag(Varyings IN) : SV_Target
        {
            half4 fragColor = float4(0, 0, 0, 0);
            float2 FragUv = IN.uv;
            float TimeRate = _Speed; //旋转速度
            float a0 = 0.0; // 无量纲自旋系数
            const float Rs1 = _Mass * 2.952764e3; //这里的值得手动算好，直接计算会因为精度问题导致错误  实际公式2 * _Mass * GRAVITY_CONSTANT / SPEED_OF_LIGHT / SPEED_OF_LIGHT * SOLAR_MASS
            float z1 = 1 + pow(1. - a0 * a0, 0.333333333333333) * (pow(1. + a0 * a0, 0.333333333333333) + pow(1. - a0, 0.333333333333333)); // 辅助变量
            float RmsRatio = (3 + sqrt(3 * a0 * a0 + z1 * z1) - sqrt((3. - z1) * (3. + z1 + 2 * sqrt(3. * a0 * a0 + z1 * z1)))) / 2; // 赤道顺行最内稳定圆轨与Rs之比
            float AccEff = sqrt(1. - 1. / RmsRatio); // 吸积放能效率,以落到Rms为准
            float mu = 1; // 吸积物的比荷的倒数,氕为1
            float dmdtEdd = 6.327 * mu / SPEED_OF_LIGHT / SPEED_OF_LIGHT * _Mass * SOLAR_MASS / AccEff; // 爱丁顿吸积率
            float dmdt = _DmdtMulti * dmdtEdd; // 吸积率
            float diskA = 3. * GRAVITY_CONSTANT * SOLAR_MASS / Rs1 / Rs1 / Rs1 * _Mass * dmdt / (8. * PI * SIGMA); // 吸积盘温度系数

            float QuadraticedPeakTemperature = diskA * 0.05665278; // 计算峰值温度的四次方,用于自适应亮度。峰值温度出现在49InterRadius/36处
            const float Rs = Rs1 / LIGHT_YEAR; // 单位是ly
            const float InterRadius = 0.7 * RmsRatio * Rs; // 盘内缘,正常情况下等于最内稳定圆轨
            const float OuterRadius = _Radius * Rs; // 盘外缘
            //float3 WorldUp = GetCameraRot(float4(0, 1, 0, 0));

            // 以下在相机系
            float3 BlackHoleRPos = GetCameraTransform(float4(_BlackHolePos, 1)) * Rs;
            float3 BlackHoleRDiskNormal = GetCameraRot(float4(_BlackHoleUp, 0));
            float3 BlackHoleRDiskForward = GetCameraRot(float4(_BlackHoleForward, 0));
            float3 RayDir = UvToDir(FragUv + 0.5 * float2(RandomStep(FragUv, frac(_Time.y + 0.5)), RandomStep(FragUv, frac(_Time.y))) / _Resolution.xy);

            float3 RayPos = float3(0, 0, 0);
            // float3 LastRayPos;
            // float3 LastRayDir;
            float3 PosToBlackHole = RayPos - BlackHoleRPos;
            float StepLength = 0;
            float LastR = length(PosToBlackHole);
            uint Count = 0;
            
            [loop]
            for (uint i = 0; i < _Iterations && fragColor.a < 0.99; i++)
            {
                PosToBlackHole = RayPos - BlackHoleRPos;
                float DistanceToBlackHole = length(PosToBlackHole);
                float3 NormalizedPosToBlackHole = PosToBlackHole / DistanceToBlackHole;
                //float dotWithNormal = dot(BlackHoleRDiskNormal, PosToBlackHole);
                if (DistanceToBlackHole > 2.5 * OuterRadius && DistanceToBlackHole > LastR && Count > 50)
                {
                    // FragUv = DirToUV(RayDir);
                    // int2 coord1 = int2(frac(FragUv.xy) * float2(512, 512));
                    // float4 fetched_color = _MainTex.Load(int3(coord1, 0));
                    // fragColor += 0.5 * fetched_color * (1.0 - fragColor.a);
                    break;
                }

                if (DistanceToBlackHole < 0.1 * Rs)
                {
                    // 命中黑洞
                    break;
                }

                fragColor = DiskColor(fragColor, TimeRate, StepLength, RayPos, RayDir, BlackHoleRPos, BlackHoleRDiskForward, BlackHoleRDiskNormal,
                                            Rs, InterRadius, OuterRadius, diskA, QuadraticedPeakTemperature, SHIFT_MAX);
                // LastRayPos = RayPos;
                // LastRayDir = RayDir;
                LastR = DistanceToBlackHole;
                float CosTheta = length(cross(NormalizedPosToBlackHole, RayDir));
                float DeltaPhiRate = -CosTheta * CosTheta * CosTheta * (1.5 * Rs / DistanceToBlackHole);

                float RayStep = Count == 0 ? RandomStep(FragUv, frac(_Time.y)) : 1.0;
                RayStep *= 0.15 + 0.25 * saturate(0.5 * (0.5 * DistanceToBlackHole / max(10.0 * Rs, OuterRadius) - 1.0));

                //步长计算
                if (DistanceToBlackHole >= 2.0 * OuterRadius)
                {
                    RayStep *= DistanceToBlackHole;
                }
                else if (DistanceToBlackHole >= 1.0 * OuterRadius)
                {
                    RayStep *= (Rs * (2.0 * OuterRadius - DistanceToBlackHole) + DistanceToBlackHole * (DistanceToBlackHole - OuterRadius)) / OuterRadius;
                }
                else
                {
                    RayStep *= min(Rs, DistanceToBlackHole);
                }

                RayPos += RayDir * RayStep;
                float DeltaPhi = RayStep / DistanceToBlackHole * DeltaPhiRate;
                RayDir = normalize(RayDir + (DeltaPhi + DeltaPhi * DeltaPhi * DeltaPhi / 3.0) * cross(cross(RayDir, NormalizedPosToBlackHole), RayDir) / CosTheta);
                StepLength = RayStep;
                Count++;
            }

            //为了套bloom先逆处理一遍
            float colorRFactor = fragColor.r / fragColor.g;
            float colorBFactor = fragColor.b / fragColor.g;
            
            float bloomMax = _Bloom;
            fragColor.r = min(-4.0 * log(1. - pow(fragColor.r, 2.2)), bloomMax * colorRFactor);
            fragColor.g = min(-4.0 * log(1. - pow(fragColor.g, 2.2)), bloomMax);
            fragColor.b = min(-4.0 * log(1. - pow(fragColor.b, 2.2)), bloomMax * colorBFactor);
            fragColor.a = min(-4.0 * log(1. - pow(fragColor.a, 2.2)), 4.0);

            // TAA 换成Unity2022自带的
            // float blendWeight = 1.0 - pow(
            //     0.5, unity_DeltaTime / max(min(0.131 * 36.0 / _Speed * GetKeplerianAngularVelocity(3 * 0.00000465, 0.00000465) / GetKeplerianAngularVelocity(3 * Rs, Rs), 0.3), 0.02));
            // return lerp(SAMPLE_TEXTURE2D(_TempTexture, sampler_TempTexture, IN.uv), fragColor, blendWeight);
            return fragColor;
        }
        ENDHLSL

        Pass
        {
            Name "BlackHolePass"
            ZTest Always
            ZWrite Off
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            ENDHLSL
        }
    }
}