#ifndef MAXWELL_FDTD_KERNELS_SPECTRAL_SLANG
#define MAXWELL_FDTD_KERNELS_SPECTRAL_SLANG

[AutoPyBindCUDA]
[CUDAKernel]
void accumulateRunningDftYee3D(
    TensorView<float> Ex,
    TensorView<float> Ey,
    TensorView<float> Ez,
    TensorView<float> ExRealAccum,
    TensorView<float> ExImagAccum,
    TensorView<float> EyRealAccum,
    TensorView<float> EyImagAccum,
    TensorView<float> EzRealAccum,
    TensorView<float> EzImagAccum,
    float weightedCos,
    float weightedSin
)
{
    uint linearIndex = linearThreadIndex1D();

    uint exTotal = linearSize3D(Ex);
    if (linearIndex < exTotal)
    {
        uint3 exIndex = unflatten3D(linearIndex, Ex.size(1), Ex.size(2));
        float exValue = Ex[exIndex.x, exIndex.y, exIndex.z];
        ExRealAccum[exIndex.x, exIndex.y, exIndex.z] += exValue * weightedCos;
        ExImagAccum[exIndex.x, exIndex.y, exIndex.z] += exValue * weightedSin;
    }

    uint eyTotal = linearSize3D(Ey);
    if (linearIndex < eyTotal)
    {
        uint3 eyIndex = unflatten3D(linearIndex, Ey.size(1), Ey.size(2));
        float eyValue = Ey[eyIndex.x, eyIndex.y, eyIndex.z];
        EyRealAccum[eyIndex.x, eyIndex.y, eyIndex.z] += eyValue * weightedCos;
        EyImagAccum[eyIndex.x, eyIndex.y, eyIndex.z] += eyValue * weightedSin;
    }

    uint ezTotal = linearSize3D(Ez);
    if (linearIndex < ezTotal)
    {
        uint3 ezIndex = unflatten3D(linearIndex, Ez.size(1), Ez.size(2));
        float ezValue = Ez[ezIndex.x, ezIndex.y, ezIndex.z];
        EzRealAccum[ezIndex.x, ezIndex.y, ezIndex.z] += ezValue * weightedCos;
        EzImagAccum[ezIndex.x, ezIndex.y, ezIndex.z] += ezValue * weightedSin;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void accumulateRunningDftYee3DBatched(
    TensorView<float> Ex,
    TensorView<float> Ey,
    TensorView<float> Ez,
    TensorView<float> ExRealAccum,
    TensorView<float> ExImagAccum,
    TensorView<float> EyRealAccum,
    TensorView<float> EyImagAccum,
    TensorView<float> EzRealAccum,
    TensorView<float> EzImagAccum,
    TensorView<float> weightedCos,
    TensorView<float> weightedSin
)
{
    uint linearIndex = linearThreadIndex1D();
    uint frequencyCount = weightedCos.size(0);

    uint exElements = linearSize3D(Ex);
    uint exTotal = exElements * frequencyCount;
    if (linearIndex < exTotal)
    {
        uint frequencyIndex = linearIndex / exElements;
        uint exLinearIndex = linearIndex - frequencyIndex * exElements;
        uint3 exIndex = unflatten3D(exLinearIndex, Ex.size(1), Ex.size(2));
        float exValue = Ex[exIndex.x, exIndex.y, exIndex.z];
        float weightCos = weightedCos[frequencyIndex];
        float weightSin = weightedSin[frequencyIndex];
        ExRealAccum[frequencyIndex, exIndex.x, exIndex.y, exIndex.z] += exValue * weightCos;
        ExImagAccum[frequencyIndex, exIndex.x, exIndex.y, exIndex.z] += exValue * weightSin;
    }

    uint eyElements = linearSize3D(Ey);
    uint eyTotal = eyElements * frequencyCount;
    if (linearIndex < eyTotal)
    {
        uint frequencyIndex = linearIndex / eyElements;
        uint eyLinearIndex = linearIndex - frequencyIndex * eyElements;
        uint3 eyIndex = unflatten3D(eyLinearIndex, Ey.size(1), Ey.size(2));
        float eyValue = Ey[eyIndex.x, eyIndex.y, eyIndex.z];
        float weightCos = weightedCos[frequencyIndex];
        float weightSin = weightedSin[frequencyIndex];
        EyRealAccum[frequencyIndex, eyIndex.x, eyIndex.y, eyIndex.z] += eyValue * weightCos;
        EyImagAccum[frequencyIndex, eyIndex.x, eyIndex.y, eyIndex.z] += eyValue * weightSin;
    }

    uint ezElements = linearSize3D(Ez);
    uint ezTotal = ezElements * frequencyCount;
    if (linearIndex < ezTotal)
    {
        uint frequencyIndex = linearIndex / ezElements;
        uint ezLinearIndex = linearIndex - frequencyIndex * ezElements;
        uint3 ezIndex = unflatten3D(ezLinearIndex, Ez.size(1), Ez.size(2));
        float ezValue = Ez[ezIndex.x, ezIndex.y, ezIndex.z];
        float weightCos = weightedCos[frequencyIndex];
        float weightSin = weightedSin[frequencyIndex];
        EzRealAccum[frequencyIndex, ezIndex.x, ezIndex.y, ezIndex.z] += ezValue * weightCos;
        EzImagAccum[frequencyIndex, ezIndex.x, ezIndex.y, ezIndex.z] += ezValue * weightSin;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void accumulatePointObservers3D(
    TensorView<float> field,
    TensorView<int> pointI,
    TensorView<int> pointJ,
    TensorView<int> pointK,
    TensorView<float> realAccum,
    TensorView<float> imagAccum,
    float weightedCos,
    float weightedSin
)
{
    uint linearIndex = linearThreadIndex1D();
    if (linearIndex >= realAccum.size(0))
    {
        return;
    }

    uint i = uint(pointI[linearIndex]);
    uint j = uint(pointJ[linearIndex]);
    uint k = uint(pointK[linearIndex]);
    float value = field[i, j, k];
    realAccum[linearIndex] += value * weightedCos;
    imagAccum[linearIndex] += value * weightedSin;
}

[AutoPyBindCUDA]
[CUDAKernel]
void accumulatePlaneObserver3D(
    TensorView<float> field,
    TensorView<float> planeRealAccum,
    TensorView<float> planeImagAccum,
    uint axisCode,
    uint planeIndex,
    float weightedCos,
    float weightedSin
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = planeRealAccum.size(0) * planeRealAccum.size(1);
    if (linearIndex >= total)
    {
        return;
    }

    uint u = linearIndex / planeRealAccum.size(1);
    uint v = linearIndex - u * planeRealAccum.size(1);
    float value = 0.0;
    if (axisCode == 0)
    {
        value = field[planeIndex, u, v];
    }
    else if (axisCode == 1)
    {
        value = field[u, planeIndex, v];
    }
    else
    {
        value = field[u, v, planeIndex];
    }

    planeRealAccum[u, v] += value * weightedCos;
    planeImagAccum[u, v] += value * weightedSin;
}

#endif
