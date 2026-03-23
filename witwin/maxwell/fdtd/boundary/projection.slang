#ifndef MAXWELL_FDTD_BOUNDARY_PROJECTION_SLANG
#define MAXWELL_FDTD_BOUNDARY_PROJECTION_SLANG

#include "helpers.slang"

[AutoPyBindCUDA]
[CUDAKernel]
void clampPecBoundary3D(
    TensorView<float> field,
    uint axisA,
    uint axisB
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(field);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, field.size(1), field.size(2));
    if (isAxisBoundaryIndex(field, index, axisA) || isAxisBoundaryIndex(field, index, axisB))
    {
        field[index.x, index.y, index.z] = 0.0;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void projectPeriodicBoundary3D(
    TensorView<float> field,
    uint axis
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = faceLinearSize(field, axis);
    if (linearIndex >= total)
    {
        return;
    }

    if (axis == 0)
    {
        uint j = linearIndex / field.size(2);
        uint k = linearIndex - j * field.size(2);
        uint highI = field.size(0) - 1;
        float averageValue = 0.5 * (field[0, j, k] + field[highI, j, k]);
        field[0, j, k] = averageValue;
        field[highI, j, k] = averageValue;
        return;
    }

    if (axis == 1)
    {
        uint i = linearIndex / field.size(2);
        uint k = linearIndex - i * field.size(2);
        uint highJ = field.size(1) - 1;
        float averageValue = 0.5 * (field[i, 0, k] + field[i, highJ, k]);
        field[i, 0, k] = averageValue;
        field[i, highJ, k] = averageValue;
        return;
    }

    uint i = linearIndex / field.size(1);
    uint j = linearIndex - i * field.size(1);
    uint highK = field.size(2) - 1;
    float averageValue = 0.5 * (field[i, j, 0] + field[i, j, highK]);
    field[i, j, 0] = averageValue;
    field[i, j, highK] = averageValue;
}

[AutoPyBindCUDA]
[CUDAKernel]
void projectBlochBoundary3D(
    TensorView<float> fieldReal,
    TensorView<float> fieldImag,
    uint axis,
    float phaseCos,
    float phaseSin
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = faceLinearSize(fieldReal, axis);
    if (linearIndex >= total)
    {
        return;
    }

    if (axis == 0)
    {
        uint j = linearIndex / fieldReal.size(2);
        uint k = linearIndex - j * fieldReal.size(2);
        uint highI = fieldReal.size(0) - 1;
        float2 low = float2(fieldReal[0, j, k], fieldImag[0, j, k]);
        float2 high = float2(fieldReal[highI, j, k], fieldImag[highI, j, k]);
        float2 projectedLow = 0.5 * (low + complexPhaseNegative(phaseCos, phaseSin, high));
        float2 projectedHigh = complexPhasePositive(phaseCos, phaseSin, projectedLow);
        fieldReal[0, j, k] = projectedLow.x;
        fieldImag[0, j, k] = projectedLow.y;
        fieldReal[highI, j, k] = projectedHigh.x;
        fieldImag[highI, j, k] = projectedHigh.y;
        return;
    }

    if (axis == 1)
    {
        uint i = linearIndex / fieldReal.size(2);
        uint k = linearIndex - i * fieldReal.size(2);
        uint highJ = fieldReal.size(1) - 1;
        float2 low = float2(fieldReal[i, 0, k], fieldImag[i, 0, k]);
        float2 high = float2(fieldReal[i, highJ, k], fieldImag[i, highJ, k]);
        float2 projectedLow = 0.5 * (low + complexPhaseNegative(phaseCos, phaseSin, high));
        float2 projectedHigh = complexPhasePositive(phaseCos, phaseSin, projectedLow);
        fieldReal[i, 0, k] = projectedLow.x;
        fieldImag[i, 0, k] = projectedLow.y;
        fieldReal[i, highJ, k] = projectedHigh.x;
        fieldImag[i, highJ, k] = projectedHigh.y;
        return;
    }

    uint i = linearIndex / fieldReal.size(1);
    uint j = linearIndex - i * fieldReal.size(1);
    uint highK = fieldReal.size(2) - 1;
    float2 low = float2(fieldReal[i, j, 0], fieldImag[i, j, 0]);
    float2 high = float2(fieldReal[i, j, highK], fieldImag[i, j, highK]);
    float2 projectedLow = 0.5 * (low + complexPhaseNegative(phaseCos, phaseSin, high));
    float2 projectedHigh = complexPhasePositive(phaseCos, phaseSin, projectedLow);
    fieldReal[i, j, 0] = projectedLow.x;
    fieldImag[i, j, 0] = projectedLow.y;
    fieldReal[i, j, highK] = projectedHigh.x;
    fieldImag[i, j, highK] = projectedHigh.y;
}

#endif
