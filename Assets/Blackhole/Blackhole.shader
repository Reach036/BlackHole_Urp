Shader "Custom/Blackhole"
{
    Properties
    {
        //[MainColor] _BaseColor("Base Color", Color) = (1, 1, 1, 1)
        //[MainTexture] _BaseMap("Base Map", 2D) = "white"
        //_MainTex ("Main Texture", 2D) = "white" {}
        _Radius ("吸积盘半径", Float) = 12
        _Mass ("黑洞质量（单位为一个太阳质量）", Float) = 14900000
        _Speed ("黑洞转速", int) = 30
        _Iterations ("迭代次数", int) = 150
        _DmdtMulti ("吸积率", Range(0.00000001, 0.1)) = 0.000002
        _NoiseLUT("Noise LUT", 2D) = "white" {}
    }

    SubShader
    {
        Tags
        {
            "RenderPipeline" = "UniversalPipeline" "RenderType" = "Transparent" "Queue" = "Transparent"
        }
        Blend SrcAlpha OneMinusSrcAlpha

        Cull Off ZWrite Off ZTest Always

        Pass
        {
            HLSLPROGRAM
            //#pragma enable_d3d11_debug_symbols
            #pragma target 5.0
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "3DNoiseCore.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float _Radius;
                float _Mass;
                float _DmdtMulti;
                uint _Iterations;
                int _Speed;
            CBUFFER_END

            #define SPEED_OF_LIGHT 2.99792458e8
            #define LIGHT_YEAR 9.4607e15
            #define GRAVITY_CONSTANT 6.67430e-11
            #define SOLAR_MASS 1.98847e30
            #define SIGMA 5.670373e-8
            #define SHIFT_MAX 1.25
            #define RANDOM_SEED 43758.5453


            inline float RandomStep(float2 xy, float seed)
            {
                return frac(sin(dot(xy + frac(11.4514 * sin(seed)), float2(12.9898, 78.233))) * RANDOM_SEED);
            }

            float SoftSaturate(float x)
            {
                return 1.0 - 1.0 / (max(x, 0.0) + 1.0);
            }

            float GenerateAccretionDiskNoise(float3 Position, int NoiseStartLevel, int NoiseEndLevel, float ContrastLevel)
            {
                float NoiseAccumulator = 10.0;
                [unroll]
                for (int Level = NoiseStartLevel; Level < NoiseEndLevel; ++Level) {
                    NoiseAccumulator *= 1.0 + 0.1 * noise(float3(Position) * pow(3.0, float(Level)));
                }
                NoiseAccumulator = abs(NoiseAccumulator);
                return log(1.0 + pow(0.1 * NoiseAccumulator, ContrastLevel));
            }

            float Vec2ToTheta(float2 v1, float2 v2)
            {
                float dotProduct = dot(v1, v2);
                float crossProduct = v1.x * v2.y - v1.y * v2.x;
                float angle = asin(0.999999 * crossProduct / (length(v1) * length(v2)));
                if (dotProduct > 0.0)
                    return angle;
                if (crossProduct < 0.0)
                    return PI - angle;
                return -PI - angle;
            }

            float3 KelvinToRgb(float Kelvin)
            {
                if (Kelvin < 400.01) {
                    return float3(0, 0, 0);
                }

                float Teff = (Kelvin - 6500.0) / (6500.0 * Kelvin * 2.2);
                float3 RgbColor = float3(0, 0, 0);

                RgbColor.r = exp(2.05539304e4 * Teff);
                RgbColor.g = exp(2.63463675e4 * Teff);
                RgbColor.b = exp(3.30145739e4 * Teff);

                float BrightnessScale = 1.0 / max(max(RgbColor.r, RgbColor.g), RgbColor.b);

                if (Kelvin < 1000.0) {
                    BrightnessScale *= (Kelvin - 400.0) / 600.0;
                }

                RgbColor *= BrightnessScale;
                return RgbColor;
            }

            float GetKeplerianAngularVelocity(float Radius, float Rs)
            {
                return sqrt(SPEED_OF_LIGHT / LIGHT_YEAR * SPEED_OF_LIGHT * Rs / LIGHT_YEAR / ((2.0 * Radius - 3.0 * Rs) * Radius * Radius));
            }

            float Shape(float x, float Alpha, float Beta)
            {
                float k = pow(Alpha + Beta, Alpha + Beta) / (pow(Alpha, Alpha) * pow(Beta, Beta));
                return k * pow(x, Alpha) * pow(abs(1.0 - x), Beta);
            }

            float4 DiskColor(float4 BaseColor, float TimeRate, float StepLength, float3 RayPos,
                                float3 RayDir, float Rs, float InterRadius, float OuterRadius,
                                float DiskTemperatureArgument, float QuadraticedPeakTemperature, float ShiftMax, float someCameraParam)
            {
                float3 PosOnDisk = RayPos;
                float3 DirOnDisk = RayDir;

                float PosR = length(PosOnDisk.xz);
                float PosY = PosOnDisk.y;

                float4 Color = float4(0.0, 0.0, 0.0, 0.0);
                if (abs(PosY) < 0.5 * Rs && PosR < OuterRadius && PosR > InterRadius) {
                    float EffectiveRadius = 1.0 - (PosR - InterRadius) / (OuterRadius - InterRadius) * 0.5;
                    if (OuterRadius - InterRadius > 9.0 * Rs) {
                        if (PosR < 5.0 * Rs + InterRadius) {
                            EffectiveRadius = 1.0 - (PosR - InterRadius) / (9.0 * Rs) * 0.5;
                        }
                        else {
                            EffectiveRadius = 1.0 - (0.5 / 0.9 * 0.5 + ((PosR - InterRadius) / (OuterRadius - InterRadius) -
                                5.0 * Rs / (OuterRadius - InterRadius)) / (1.0 - 5.0 * Rs / (OuterRadius - InterRadius)) * 0.5);
                        }
                    }

                    if (abs(PosY) < 0.5 * Rs * Shape(EffectiveRadius, 4.0, 0.9) || PosY < 0.5 * Rs * (1.0 - 5.0 * pow(2.0 * (1.0 - EffectiveRadius), 2.0))) {
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
                        float RedShift = Dopler * sqrt(max(1.0 - Rs / PosR, 0.000001)) * someCameraParam;

                        float Thick;
                        float VerticalMixFactor;
                        float DustColor = 0.0;
                        float4 Color0 = float4(0.0, 0.0, 0.0, 0.0);
                        float4 Color1 = float4(0.0, 0.0, 0.0, 0.0);
                        float Density = Shape(EffectiveRadius, 4.0, 0.9);

                        float tempY = 0.5 * Rs * Density;
                        if (abs(PosY) < tempY) {
                            Thick = tempY * (0.4 + 0.6 * SoftSaturate(GenerateAccretionDiskNoise(float3(1.5 * PosTheta, PosR / Rs, 1.0), 1, 3, 80.0))); // 盘厚
                            VerticalMixFactor = max(0.0, 1.0 - abs(PosY) / Thick);
                            Density *= 0.7 * VerticalMixFactor * Density;

                            float tempNoise = GenerateAccretionDiskNoise(float3(1.0 * PosR / Rs, 1.0 * PosY / Rs, 0.5 * PosTheta), 3, 6, 80.0);
                            Color0 = float4(tempNoise, tempNoise, tempNoise, tempNoise); // 云本体
                            Color0.xyz *= Density * 1.4 * (0.2 + 0.8 * VerticalMixFactor + (0.8 - 0.8 * VerticalMixFactor) *
                                GenerateAccretionDiskNoise(float3(PosR / Rs, 1.5 * PosTheta, PosY / Rs), 1, 3, 80.0));
                            Color0.a *= Density; // * (1.0 + VerticalMixFactor);
                        }
                        if (abs(PosY) < 0.5 * Rs * (1.0 - 5.0 * pow(2.0 * (1.0 - EffectiveRadius), 2.0))) {
                            DustColor = max(1.0 - pow(PosY / (0.5 * Rs * max(1.0 - 5.0 * pow(2.0 * (1.0 - EffectiveRadius), 2.0), 0.0001)), 2.0), 0.0) * GenerateAccretionDiskNoise(
                                float3(1.5 * frac((PosThetaWithoutTime + PI / HalfPiTimeInside * EffectiveTime0 + Phase0) / (2.0 * PI)) * 2.0 * PI, PosR / Rs, PosY / Rs), 0, 6, 80.0);
                            Color0 += 0.02 * float4(DustColor, DustColor, DustColor, 0.2 * DustColor) * sqrt(1.0001 - DirOnDisk.y * DirOnDisk.y) * min(1.0, Dopler * Dopler);
                        }
                        Color0 *= 0.5 - 0.5 * cos(2.0 * PI * frac(_Time.y * TimeRate / HalfPiTimeInside)); // 用于过渡

                        PosTheta = frac((PosThetaWithoutTime + AngularVelocity * EffectiveTime1 + Phase1) / (2.0 * PI)) * 2.0 * PI; // 更新相位
                        Density = Shape(EffectiveRadius, 4.0, 0.9);

                        tempY = 0.5 * Rs * Density;
                        if (abs(PosY) < tempY) {
                            Thick = tempY * (0.4 + 0.6 * SoftSaturate(GenerateAccretionDiskNoise(float3(1.5 * PosTheta, PosR / Rs, 1.0), 1, 3, 80.0)));
                            VerticalMixFactor = max(0.0, 1.0 - abs(PosY) / Thick);
                            Density *= 0.7 * VerticalMixFactor * Density;

                            float tempNoise = GenerateAccretionDiskNoise(float3(1.0 * PosR / Rs, 1.0 * PosY / Rs, 0.5 * PosTheta), 3, 6, 80.0);
                            Color1 = float4(tempNoise, tempNoise, tempNoise, tempNoise);
                            Color1.xyz *= Density * 1.4 * (0.2 + 0.8 * VerticalMixFactor + (0.8 - 0.8 * VerticalMixFactor) * GenerateAccretionDiskNoise(
                                float3(PosR / Rs, 1.5 * PosTheta, PosY / Rs), 1, 3, 80.0));
                            Color1.a *= Density; // * (1.0 + VerticalMixFactor);
                        }
                        if (abs(PosY) < 0.5 * Rs * (1.0 - 5.0 * pow(2.0 * (1.0 - EffectiveRadius), 2.0))) {
                            DustColor = max(1.0 - pow(PosY / (0.5 * Rs * max(1.0 - 5.0 * pow(2.0 * (1.0 - EffectiveRadius), 2.0), 0.0001)), 2.0), 0.0) * GenerateAccretionDiskNoise(
                                float3(1.5 * frac((PosThetaWithoutTime + PI / HalfPiTimeInside * EffectiveTime1 + Phase1) / (2.0 * PI)) * 2.0 * PI, PosR / Rs, PosY / Rs), 0, 6, 80.0);
                            Color1 += 0.02 * float4(DustColor, DustColor, DustColor, 0.2 * DustColor) * sqrt(1.0001 - DirOnDisk.y * DirOnDisk.y) * min(1.0, Dopler * Dopler);
                        }
                        Color1 *= 0.5 - 0.5 * cos(2.0 * PI * frac(_Time.y * TimeRate / HalfPiTimeInside + 0.5));

                        Color = Color1 + Color0;
                        Color *= 1.0 + 20.0 * exp(-10.0 * (PosR - InterRadius) / (OuterRadius - InterRadius)); // 内侧增加密度

                        float BrightWithoutRedshift = 4.5 * DiskTemperature * DiskTemperature * DiskTemperature * DiskTemperature / QuadraticedPeakTemperature; // 原亮度
                        if (DiskTemperature > 1000.0) {
                            DiskTemperature = max(1000.0, DiskTemperature * RedShift * Dopler * Dopler);
                        }

                        DiskTemperature = min(100000.0, DiskTemperature);

                        Color.xyz *= BrightWithoutRedshift * min(1.0, 1.8 * (OuterRadius - PosR) / (OuterRadius - InterRadius)) * KelvinToRgb(
                            DiskTemperature / exp((PosR - InterRadius) / (0.6 * (OuterRadius - InterRadius))));
                        Color.xyz *= min(ShiftMax, RedShift) * min(ShiftMax, Dopler);

                        RedShift = min(RedShift, ShiftMax);
                        float f = 1.0 - (1.0 - min(1.0, RedShift)) * (PosR - InterRadius) / (OuterRadius - InterRadius);
                        Color.xyz *= pow(abs(f), 9.0);
                        Color.xyz *= min(1.0, 1.0 + 0.5 * ((PosR - InterRadius) / InterRadius + InterRadius / (PosR - InterRadius)) - max(1.0, RedShift));

                        Color *= StepLength / Rs;
                    }
                }
                return BaseColor + Color * (1.0 - BaseColor.a);
            }

            float4 vert(uint id : SV_VertexID, out float3 dirUnorm : TexCoord0) : SV_POSITION
            {
                float4 hclipPos = GetFullScreenTriangleVertexPosition(id);
                dirUnorm = float3(hclipPos.xy / UNITY_MATRIX_P._11_22, -1);
                dirUnorm = TransformViewToWorldDir(dirUnorm, false);
                dirUnorm = TransformWorldToObjectDir(dirUnorm, false);
                return hclipPos;
            }

            half4 frag(float4 positionHCS : SV_POSITION, float3 dirUnorm : TexCoord0) : SV_Target
            {
                float4 fragColor = float4(0, 0, 0, 0);
                float TimeRate = _Speed; //旋转速度
                float a0 = 0.0; // 无量纲自旋系数                                                                          // 无量纲自旋系数 本部分在实际使用时uniform输入
                float Rs = _Mass * (2. * GRAVITY_CONSTANT / SPEED_OF_LIGHT / SPEED_OF_LIGHT * SOLAR_MASS); // 单位是米 本部分在实际使用时uniform输入
                float z1 = 1 + pow(1. - a0 * a0, 0.333333333333333) * (pow(1. + a0 * a0, 0.333333333333333) + pow(1. - a0, 0.333333333333333)); // 辅助变量
                float RmsRatio = (3 + sqrt(3 * a0 * a0 + z1 * z1) - sqrt((3. - z1) * (3. + z1 + 2 * sqrt(3. * a0 * a0 + z1 * z1)))) / 2; // 赤道顺行最内稳定圆轨与Rs之比
                float AccEff = sqrt(1. - 1. / RmsRatio); // 吸积放能效率,以落到Rms为准
                float mu = 1; // 吸积物的比荷的倒数,氕为1
                float dmdtEdd = 6.327 * mu / SPEED_OF_LIGHT / SPEED_OF_LIGHT * _Mass * SOLAR_MASS / AccEff; // 爱丁顿吸积率
                float dmdt = _DmdtMulti * dmdtEdd; // 吸积率
                float diskA = 3. * GRAVITY_CONSTANT * SOLAR_MASS / Rs / Rs / Rs * _Mass * dmdt / (8. * PI * SIGMA); // 吸积盘温度系数

                float QuadraticedPeakTemperature = diskA * 0.05665278; // 计算峰值温度的四次方,用于自适应亮度。峰值温度出现在49InterRadius/36处
                Rs = Rs / LIGHT_YEAR; // 单位是ly
                float InterRadius = 0.7 * RmsRatio * Rs; // 盘内缘,正常情况下等于最内稳定圆轨
                float OuterRadius = _Radius * Rs; // 盘外缘

                float3 cameraRPos = TransformWorldToObject(_WorldSpaceCameraPos) * Rs;
                float someCameraParam = 1 / sqrt(max(1.0 - Rs / length(cameraRPos), 0.000001));
                float3 RayPos = cameraRPos;
                float3 RayDir = normalize(dirUnorm);

                float3 BlackHoleRPos = 0;
                float3 PosToBlackHole = RayPos - BlackHoleRPos;
                float StepLength = 0;
                float LastR = length(PosToBlackHole);
                uint Count = 0;
                float RayStep = Count == 0 ? RandomStep(positionHCS.xy, frac(_Time.y)) : 1.0;

                for (uint i = 0; i < _Iterations && fragColor.a < 0.99; i++) {
                    PosToBlackHole = RayPos - BlackHoleRPos;
                    float DistanceToBlackHole = length(PosToBlackHole);
                    float3 NormalizedPosToBlackHole = PosToBlackHole / DistanceToBlackHole;
                    //float dotWithNormal = dot(BlackHoleRDiskNormal, PosToBlackHole);

                    // 逃逸
                    bool a = DistanceToBlackHole > 2.5 * OuterRadius && DistanceToBlackHole > LastR && Count > 50;
                    if (a) {
                        // FragUv = DirToUV(RayDir);
                        // int2 coord1 = int2(frac(FragUv.xy) * float2(512, 512));
                        // float4 fetched_color = _MainTex.Load(int3(coord1, 0));
                        // fragColor += 0.5 * fetched_color * (1.0 - fragColor.a);
                        break;
                    }

                    // 进入
                    bool b = DistanceToBlackHole < 0.1 * Rs;
                    if (b) {
                        //fragColor.a = 1;
                        break;
                    }

                    fragColor = DiskColor(fragColor, TimeRate, StepLength, RayPos, RayDir, Rs, InterRadius, OuterRadius, diskA, QuadraticedPeakTemperature, SHIFT_MAX, someCameraParam);

                    LastR = DistanceToBlackHole;
                    float CosTheta = length(cross(NormalizedPosToBlackHole, RayDir));
                    float DeltaPhiRate = -CosTheta * CosTheta * CosTheta * (1.5 * Rs / DistanceToBlackHole);

                    if (Count != 0)
                        RayStep = 1.0;
                    RayStep *= 0.15 + 0.25 * saturate(0.5 * (0.5 * DistanceToBlackHole / max(10.0 * Rs, OuterRadius) - 1.0));

                    //步长计算
                    if (DistanceToBlackHole >= 2.0 * OuterRadius) {
                        RayStep *= DistanceToBlackHole;
                    }
                    else if (DistanceToBlackHole >= 1.0 * OuterRadius) {
                        RayStep *= (Rs * (2.0 * OuterRadius - DistanceToBlackHole) + DistanceToBlackHole * (DistanceToBlackHole - OuterRadius)) / OuterRadius;
                    }
                    else {
                        RayStep *= min(Rs, DistanceToBlackHole);
                    }

                    RayPos += RayDir * RayStep;
                    float DeltaPhi = RayStep / DistanceToBlackHole * DeltaPhiRate;
                    RayDir = normalize(RayDir + (DeltaPhi + DeltaPhi * DeltaPhi * DeltaPhi / 3.0) * cross(cross(RayDir, NormalizedPosToBlackHole), RayDir) / CosTheta);
                    StepLength = RayStep;
                    Count++;
                }
                return fragColor;
            }
            ENDHLSL
        }
        
		Pass {
			Name "SceneSelectionPass"
			Tags { "LightMode" = "SceneSelectionPass" }
			HLSLPROGRAM
			#pragma vertex Vert
			#pragma fragment Frag

			int _ObjectId;
			int _PassValue;

			float4 Vert(float4 v : POSITION) : POSITION {
				return 0;
			}

			int2 Frag(float4 v : POSITION) : SV_Target {
				return 0;
			}
			ENDHLSL
		}

		Pass {
			Name "ScenePickingPass"
			Tags { "LightMode" = "Picking" }

			HLSLPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			float4 vert() : POSITION {
				return 0;
			}
			int frag() : SV_Target {
				return 0;
			}
			ENDHLSL
		}
    }
}