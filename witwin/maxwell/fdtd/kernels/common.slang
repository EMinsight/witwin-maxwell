#ifndef MAXWELL_FDTD_KERNELS_COMMON_SLANG
#define MAXWELL_FDTD_KERNELS_COMMON_SLANG

uint linearThreadIndex1D()
{
    return cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
}

uint linearSize3D(TensorView<float> field)
{
    return field.size(0) * field.size(1) * field.size(2);
}

uint3 unflatten3D(uint linearIndex, uint sizeY, uint sizeZ)
{
    uint planeSize = sizeY * sizeZ;
    uint i = linearIndex / planeSize;
    uint remainder = linearIndex - i * planeSize;
    uint j = remainder / sizeZ;
    uint k = remainder - j * sizeZ;
    return uint3(i, j, k);
}

float evaluateSourceTimeSample(
    uint timeKind,
    float sampleTime,
    float frequency,
    float fwidth,
    float amplitude,
    float phase,
    float delay
)
{
    float twoPi = 6.283185307179586f;
    if (timeKind == 0)
    {
        return amplitude * cos(twoPi * frequency * sampleTime + phase);
    }
    if (timeKind == 1)
    {
        float sigmaT = 1.0f / max(twoPi * fwidth, 1.0e-30f);
        float tau = sampleTime - delay;
        float normalized = tau / sigmaT;
        float envelope = exp(-0.5f * normalized * normalized);
        return amplitude * envelope * cos(twoPi * frequency * tau + phase);
    }

    float tau = sampleTime - delay;
    float alpha = 3.141592653589793f * frequency * tau;
    float alphaSq = alpha * alpha;
    return amplitude * (1.0f - 2.0f * alphaSq) * exp(-alphaSq);
}

#endif
