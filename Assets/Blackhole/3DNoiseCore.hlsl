#pragma once

TEXTURE2D(_NoiseLUT);
SAMPLER(sampler_NoiseLUT);

float noise(float3 x)
{
	const float LUTRes = 256; // LUT的分辨率
	float3 f = frac(x);
	float3 p = floor(x);
	f = f * f * (3 - 2 * f);
	float2 uv = p.xy + float2(37, 17) * p.z + f.xy;
	uv = uv * (1 / LUTRes) + 0.5 / LUTRes;
	float2 rg = _NoiseLUT.SampleLevel(sampler_NoiseLUT, uv, 0);
	return lerp(rg.x, rg.y, f.z) * 2 - 1;
}

/*
float LUTNoise4Octaves(sampler2D LUT, float3 x)
{
	float f = 0.4 * noise(LUT, x);
	x *= 2;
	f += 0.3 * noise(LUT, x);
	x *= 2.01;
	f += 0.2 * noise(LUT, x);
	x *= 2.03;
	f += 0.1 * noise(LUT, x);
	x *= 1.98;
	f += 0.05 * noise(LUT, x);
	return f;
}

float LUTNoise3Octaves(sampler2D LUT, float3 x)
{
	float f = 0.4 * noise(LUT, x);
	x *= 2;
	f += 0.3 * noise(LUT, x);
	x *= 2.01;
	f += 0.2 * noise(LUT, x);
	x *= 2.03;
	f += 0.1 * noise(LUT, x);
	x *= 1.98;
	f += 0.05 * noise(LUT, x);
	return f;
}

float LUTNoise2Octaves(sampler2D LUT, float3 x)
{
	float f = 0.65 * noise(LUT, x);
	x *= 2.01;
	f += 0.35 * noise(LUT, x);
	return f;
}

void LUTNoise_float(
	sampler2D LUT,
	int Octaves,
	float shrinkPerOct,
	float intenFallPerOct,
	bool rotatePerOct,
	float3 pos,
	out float value)
{
	float3x3 m = float3x3(
		0.0, -0.80, -0.60,
		0.8, 0.36, -0.48,
		0.6, -0.48, 0.64);
	float inten = 1;
	value = 0;
	if (rotatePerOct)
		for (; Octaves > 0; Octaves--) {
			value += inten * noise(LUT, pos);
			inten *= intenFallPerOct;
			pos = mul(m, pos) * shrinkPerOct;
		}
	else
		for (; Octaves > 0; Octaves--) {
			value += inten * noise(LUT, pos);
			inten *= intenFallPerOct;
			pos *= shrinkPerOct;
		}
}
*/
