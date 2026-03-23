#ifndef MAXWELL_FDTD_BOUNDARY_HELPERS_SLANG
#define MAXWELL_FDTD_BOUNDARY_HELPERS_SLANG

static const uint BOUNDARY_NONE = 0;
static const uint BOUNDARY_PML = 1;
static const uint BOUNDARY_PERIODIC = 2;
static const uint BOUNDARY_BLOCH = 3;
static const uint BOUNDARY_PEC = 4;
static const uint BOUNDARY_PMC = 5;

bool isBoundaryIndex(uint index, uint size)
{
    return index == 0 || index + 1 >= size;
}

bool isLowBoundaryIndex(uint index)
{
    return index == 0;
}

uint selectBoundaryMode(uint lowBoundaryMode, uint highBoundaryMode, uint boundaryIndex)
{
    return isLowBoundaryIndex(boundaryIndex) ? lowBoundaryMode : highBoundaryMode;
}

bool isPecBoundaryMode(uint boundaryMode)
{
    return boundaryMode == BOUNDARY_PEC;
}

bool isInactiveBoundaryMode(uint boundaryMode)
{
    return boundaryMode == BOUNDARY_NONE || boundaryMode == BOUNDARY_PML;
}

struct DiffFloat
{
    float value;
    bool skip;
};

DiffFloat makeDiffFloat(float value, bool skip)
{
    DiffFloat result;
    result.value = value;
    result.skip = skip;
    return result;
}

float periodicBackwardDifference(float lowValue, float highValue, float invDelta)
{
    return (lowValue - highValue) * invDelta;
}

float pmcBackwardDifference(uint boundaryIndex, float lowValue, float highValue, float invDelta)
{
    return (boundaryIndex == 0) ? 2.0 * lowValue * invDelta : -2.0 * highValue * invDelta;
}

DiffFloat boundaryBackwardDiff(
    uint boundaryMode,
    uint boundaryIndex,
    float lowValue,
    float highValue,
    float invDelta
)
{
    if (boundaryMode == BOUNDARY_PERIODIC)
    {
        return makeDiffFloat(periodicBackwardDifference(lowValue, highValue, invDelta), false);
    }

    if (boundaryMode == BOUNDARY_PMC)
    {
        return makeDiffFloat(pmcBackwardDifference(boundaryIndex, lowValue, highValue, invDelta), false);
    }

    return makeDiffFloat(0.0, true);
}

float2 complexPhasePositive(float phaseCos, float phaseSin, float2 value)
{
    return float2(
        phaseCos * value.x - phaseSin * value.y,
        phaseSin * value.x + phaseCos * value.y
    );
}

float2 complexPhaseNegative(float phaseCos, float phaseSin, float2 value)
{
    return float2(
        phaseCos * value.x + phaseSin * value.y,
        phaseCos * value.y - phaseSin * value.x
    );
}

struct DiffFloat2
{
    float2 value;
    bool skip;
};

struct CompressedCpmlIndex
{
    uint index;
    bool active;
};

CompressedCpmlIndex makeCompressedCpmlIndex(uint index, bool active)
{
    CompressedCpmlIndex result;
    result.index = index;
    result.active = active;
    return result;
}

CompressedCpmlIndex resolveCompressedCpmlIndex(
    uint globalIndex,
    uint lowLength,
    uint highStart,
    uint highLength
)
{
    if (globalIndex < lowLength)
    {
        return makeCompressedCpmlIndex(globalIndex, true);
    }

    if (highLength > 0 && globalIndex >= highStart)
    {
        return makeCompressedCpmlIndex(lowLength + (globalIndex - highStart), true);
    }

    return makeCompressedCpmlIndex(0, false);
}

DiffFloat2 makeDiffFloat2(float2 value, bool skip)
{
    DiffFloat2 result;
    result.value = value;
    result.skip = skip;
    return result;
}

DiffFloat2 blochBackwardDiff(
    uint boundaryIndex,
    float2 lowValue,
    float2 highValue,
    float phaseCos,
    float phaseSin,
    float invDelta
)
{
    if (boundaryIndex == 0)
    {
        return makeDiffFloat2(
            (lowValue - complexPhaseNegative(phaseCos, phaseSin, highValue)) * invDelta,
            false
        );
    }

    return makeDiffFloat2(
        (complexPhasePositive(phaseCos, phaseSin, lowValue) - highValue) * invDelta,
        false
    );
}

uint axisSize(TensorView<float> field, uint axis)
{
    if (axis == 0)
    {
        return field.size(0);
    }

    if (axis == 1)
    {
        return field.size(1);
    }

    return field.size(2);
}

uint axisValue(uint3 index, uint axis)
{
    if (axis == 0)
    {
        return index.x;
    }

    if (axis == 1)
    {
        return index.y;
    }

    return index.z;
}

bool isAxisBoundaryIndex(TensorView<float> field, uint3 index, uint axis)
{
    return isBoundaryIndex(axisValue(index, axis), axisSize(field, axis));
}

uint faceLinearSize(TensorView<float> field, uint axis)
{
    if (axis == 0)
    {
        return field.size(1) * field.size(2);
    }

    if (axis == 1)
    {
        return field.size(0) * field.size(2);
    }

    return field.size(0) * field.size(1);
}

#endif
