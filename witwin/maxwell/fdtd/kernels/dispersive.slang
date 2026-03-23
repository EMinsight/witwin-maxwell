#ifndef MAXWELL_FDTD_KERNELS_DISPERSIVE_SLANG
#define MAXWELL_FDTD_KERNELS_DISPERSIVE_SLANG

[AutoPyBindCUDA]
[CUDAKernel]
void updateDebyeCurrent3D(
    TensorView<float> ElectricField,
    TensorView<float> Polarization,
    TensorView<float> PolarizationCurrent,
    TensorView<float> DebyeDrive,
    float decay,
    float dt
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(ElectricField);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, ElectricField.size(1), ElectricField.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float previousPolarization = Polarization[i, j, k];
    float nextPolarization = decay * previousPolarization + DebyeDrive[i, j, k] * ElectricField[i, j, k];
    Polarization[i, j, k] = nextPolarization;
    PolarizationCurrent[i, j, k] = (nextPolarization - previousPolarization) / dt;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateDrudeCurrent3D(
    TensorView<float> ElectricField,
    TensorView<float> PolarizationCurrent,
    TensorView<float> DrudeDrive,
    float decay
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(ElectricField);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, ElectricField.size(1), ElectricField.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float nextCurrent = decay * PolarizationCurrent[i, j, k] + DrudeDrive[i, j, k] * ElectricField[i, j, k];
    PolarizationCurrent[i, j, k] = nextCurrent;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateLorentzCurrent3D(
    TensorView<float> ElectricField,
    TensorView<float> Polarization,
    TensorView<float> PolarizationCurrent,
    TensorView<float> LorentzDrive,
    float decay,
    float restoring,
    float dt
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(ElectricField);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, ElectricField.size(1), ElectricField.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float nextCurrent =
        decay * PolarizationCurrent[i, j, k]
        - restoring * Polarization[i, j, k]
        + LorentzDrive[i, j, k] * ElectricField[i, j, k];
    PolarizationCurrent[i, j, k] = nextCurrent;
    Polarization[i, j, k] += dt * nextCurrent;
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyPolarizationCurrent3D(
    TensorView<float> ElectricField,
    TensorView<float> PolarizationCurrent,
    TensorView<float> InvPermittivity,
    float dt
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(ElectricField);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, ElectricField.size(1), ElectricField.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    ElectricField[i, j, k] -= dt * PolarizationCurrent[i, j, k] * InvPermittivity[i, j, k];
}

uint clampTensorIndex(int index, uint size)
{
    if (index <= 0)
    {
        return 0;
    }

    uint maxIndex = size - 1;
    uint clamped = (uint)index;
    if (clamped > maxIndex)
    {
        return maxIndex;
    }
    return clamped;
}

float sampleTensorClamped3D(TensorView<float> field, int i, int j, int k)
{
    uint ii = clampTensorIndex(i, field.size(0));
    uint jj = clampTensorIndex(j, field.size(1));
    uint kk = clampTensorIndex(k, field.size(2));
    return field[ii, jj, kk];
}

float averageExAtEySite(TensorView<float> Ex, uint i, uint j, uint k)
{
    return 0.25f * (
        sampleTensorClamped3D(Ex, int(i) - 1, int(j), int(k))
        + sampleTensorClamped3D(Ex, int(i), int(j), int(k))
        + sampleTensorClamped3D(Ex, int(i) - 1, int(j) + 1, int(k))
        + sampleTensorClamped3D(Ex, int(i), int(j) + 1, int(k))
    );
}

float averageEzAtEySite(TensorView<float> Ez, uint i, uint j, uint k)
{
    return 0.25f * (
        sampleTensorClamped3D(Ez, int(i), int(j), int(k) - 1)
        + sampleTensorClamped3D(Ez, int(i), int(j), int(k))
        + sampleTensorClamped3D(Ez, int(i), int(j) + 1, int(k) - 1)
        + sampleTensorClamped3D(Ez, int(i), int(j) + 1, int(k))
    );
}

float averageEyAtExSite(TensorView<float> Ey, uint i, uint j, uint k)
{
    return 0.25f * (
        sampleTensorClamped3D(Ey, int(i), int(j) - 1, int(k))
        + sampleTensorClamped3D(Ey, int(i), int(j), int(k))
        + sampleTensorClamped3D(Ey, int(i) + 1, int(j) - 1, int(k))
        + sampleTensorClamped3D(Ey, int(i) + 1, int(j), int(k))
    );
}

float averageEzAtExSite(TensorView<float> Ez, uint i, uint j, uint k)
{
    return 0.25f * (
        sampleTensorClamped3D(Ez, int(i), int(j), int(k) - 1)
        + sampleTensorClamped3D(Ez, int(i), int(j), int(k))
        + sampleTensorClamped3D(Ez, int(i) + 1, int(j), int(k) - 1)
        + sampleTensorClamped3D(Ez, int(i) + 1, int(j), int(k))
    );
}

float averageExAtEzSite(TensorView<float> Ex, uint i, uint j, uint k)
{
    return 0.25f * (
        sampleTensorClamped3D(Ex, int(i) - 1, int(j), int(k))
        + sampleTensorClamped3D(Ex, int(i), int(j), int(k))
        + sampleTensorClamped3D(Ex, int(i) - 1, int(j), int(k) + 1)
        + sampleTensorClamped3D(Ex, int(i), int(j), int(k) + 1)
    );
}

float averageEyAtEzSite(TensorView<float> Ey, uint i, uint j, uint k)
{
    return 0.25f * (
        sampleTensorClamped3D(Ey, int(i), int(j) - 1, int(k))
        + sampleTensorClamped3D(Ey, int(i), int(j), int(k))
        + sampleTensorClamped3D(Ey, int(i), int(j) - 1, int(k) + 1)
        + sampleTensorClamped3D(Ey, int(i), int(j), int(k) + 1)
    );
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateKerrElectricFieldExCurl3D(
    TensorView<float> DynamicCurl,
    TensorView<float> Ex,
    TensorView<float> Ey,
    TensorView<float> Ez,
    TensorView<float> LinearPermittivity,
    TensorView<float> ExDecay,
    TensorView<float> KerrChi3,
    float dt,
    float eps0
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(DynamicCurl);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, DynamicCurl.size(1), DynamicCurl.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float ex = Ex[i, j, k];
    float ey = averageEyAtExSite(Ey, i, j, k);
    float ez = averageEzAtExSite(Ez, i, j, k);
    float intensity = ex * ex + ey * ey + ez * ez;
    float effectivePermittivity = LinearPermittivity[i, j, k] + eps0 * KerrChi3[i, j, k] * intensity;
    float permittivityFloor = 1.0e-12f * eps0;
    if (effectivePermittivity < permittivityFloor)
    {
        effectivePermittivity = permittivityFloor;
    }

    DynamicCurl[i, j, k] = (dt / effectivePermittivity) * ExDecay[i, j, k];
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateKerrElectricFieldEyCurl3D(
    TensorView<float> DynamicCurl,
    TensorView<float> Ex,
    TensorView<float> Ey,
    TensorView<float> Ez,
    TensorView<float> LinearPermittivity,
    TensorView<float> EyDecay,
    TensorView<float> KerrChi3,
    float dt,
    float eps0
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(DynamicCurl);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, DynamicCurl.size(1), DynamicCurl.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float ex = averageExAtEySite(Ex, i, j, k);
    float ey = Ey[i, j, k];
    float ez = averageEzAtEySite(Ez, i, j, k);
    float intensity = ex * ex + ey * ey + ez * ez;
    float effectivePermittivity = LinearPermittivity[i, j, k] + eps0 * KerrChi3[i, j, k] * intensity;
    float permittivityFloor = 1.0e-12f * eps0;
    if (effectivePermittivity < permittivityFloor)
    {
        effectivePermittivity = permittivityFloor;
    }

    DynamicCurl[i, j, k] = (dt / effectivePermittivity) * EyDecay[i, j, k];
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateKerrElectricFieldEzCurl3D(
    TensorView<float> DynamicCurl,
    TensorView<float> Ex,
    TensorView<float> Ey,
    TensorView<float> Ez,
    TensorView<float> LinearPermittivity,
    TensorView<float> EzDecay,
    TensorView<float> KerrChi3,
    float dt,
    float eps0
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(DynamicCurl);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, DynamicCurl.size(1), DynamicCurl.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float ex = averageExAtEzSite(Ex, i, j, k);
    float ey = averageEyAtEzSite(Ey, i, j, k);
    float ez = Ez[i, j, k];
    float intensity = ex * ex + ey * ey + ez * ez;
    float effectivePermittivity = LinearPermittivity[i, j, k] + eps0 * KerrChi3[i, j, k] * intensity;
    float permittivityFloor = 1.0e-12f * eps0;
    if (effectivePermittivity < permittivityFloor)
    {
        effectivePermittivity = permittivityFloor;
    }

    DynamicCurl[i, j, k] = (dt / effectivePermittivity) * EzDecay[i, j, k];
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateAuxiliaryMagnetic1D(
    TensorView<float> Magnetic,
    TensorView<float> Electric,
    TensorView<float> MagneticDecay,
    TensorView<float> MagneticCurl
)
{
    uint linearIndex = linearThreadIndex1D();
    if (linearIndex >= Magnetic.size(0))
    {
        return;
    }

    Magnetic[linearIndex] = MagneticDecay[linearIndex] * Magnetic[linearIndex]
        - MagneticCurl[linearIndex] * (Electric[linearIndex + 1] - Electric[linearIndex]);
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateAuxiliaryElectric1D(
    TensorView<float> Electric,
    TensorView<float> Magnetic,
    TensorView<float> ElectricDecay,
    TensorView<float> ElectricCurl,
    uint sourceIndex,
    float sourceValue
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = Electric.size(0);
    if (linearIndex >= total)
    {
        return;
    }

    bool isUpperBoundary = linearIndex + 1 == total;
    if (isUpperBoundary)
    {
        Electric[linearIndex] = 0.0f;
        return;
    }

    if (linearIndex > 0 && linearIndex != sourceIndex)
    {
        float curlH = Magnetic[linearIndex] - Magnetic[linearIndex - 1];
        Electric[linearIndex] = ElectricDecay[linearIndex] * Electric[linearIndex]
            - ElectricCurl[linearIndex] * curlH;
    }

    if (linearIndex == sourceIndex)
    {
        Electric[linearIndex] = sourceValue;
    }
}

#endif
