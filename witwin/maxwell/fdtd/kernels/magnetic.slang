#ifndef MAXWELL_FDTD_KERNELS_MAGNETIC_SLANG
#define MAXWELL_FDTD_KERNELS_MAGNETIC_SLANG

[AutoPyBindCUDA]
[CUDAKernel]
void updateMagneticFieldHx3D(
    TensorView<float> Hx,
    TensorView<float> Ey,
    TensorView<float> Ez,
    TensorView<float> HxDecay,
    TensorView<float> HxCurl,
    TensorView<float> PsiHxY,
    TensorView<float> PsiHxZ,
    TensorView<float> InvKappaHxY,
    TensorView<float> ByHxY,
    TensorView<float> CyHxY,
    TensorView<float> InvKappaHxZ,
    TensorView<float> ByHxZ,
    TensorView<float> CyHxZ,
    float invDy,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hx);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hx.size(1), Hx.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float dEz_dy = (Ez[i, j + 1, k] - Ez[i, j, k]) * invDy;
    float dEy_dz = (Ey[i, j, k + 1] - Ey[i, j, k]) * invDz;
    float psiY = ByHxY[j] * PsiHxY[i, j, k] + CyHxY[j] * dEz_dy;
    float psiZ = ByHxZ[k] * PsiHxZ[i, j, k] + CyHxZ[k] * dEy_dz;
    PsiHxY[i, j, k] = psiY;
    PsiHxZ[i, j, k] = psiZ;

    float curlE = (dEz_dy * InvKappaHxY[j] + psiY) - (dEy_dz * InvKappaHxZ[k] + psiZ);
    Hx[i, j, k] = Hx[i, j, k] * HxDecay[i, j, k] - HxCurl[i, j, k] * curlE;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateMagneticFieldHy3D(
    TensorView<float> Hy,
    TensorView<float> Ex,
    TensorView<float> Ez,
    TensorView<float> HyDecay,
    TensorView<float> HyCurl,
    TensorView<float> PsiHyX,
    TensorView<float> PsiHyZ,
    TensorView<float> InvKappaHyX,
    TensorView<float> ByHyX,
    TensorView<float> CyHyX,
    TensorView<float> InvKappaHyZ,
    TensorView<float> ByHyZ,
    TensorView<float> CyHyZ,
    float invDx,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hy);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hy.size(1), Hy.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float dEx_dz = (Ex[i, j, k + 1] - Ex[i, j, k]) * invDz;
    float dEz_dx = (Ez[i + 1, j, k] - Ez[i, j, k]) * invDx;
    float psiX = ByHyX[i] * PsiHyX[i, j, k] + CyHyX[i] * dEz_dx;
    float psiZ = ByHyZ[k] * PsiHyZ[i, j, k] + CyHyZ[k] * dEx_dz;
    PsiHyX[i, j, k] = psiX;
    PsiHyZ[i, j, k] = psiZ;

    float curlE = (dEx_dz * InvKappaHyZ[k] + psiZ) - (dEz_dx * InvKappaHyX[i] + psiX);
    Hy[i, j, k] = Hy[i, j, k] * HyDecay[i, j, k] - HyCurl[i, j, k] * curlE;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateMagneticFieldHz3D(
    TensorView<float> Hz,
    TensorView<float> Ex,
    TensorView<float> Ey,
    TensorView<float> HzDecay,
    TensorView<float> HzCurl,
    TensorView<float> PsiHzX,
    TensorView<float> PsiHzY,
    TensorView<float> InvKappaHzX,
    TensorView<float> ByHzX,
    TensorView<float> CyHzX,
    TensorView<float> InvKappaHzY,
    TensorView<float> ByHzY,
    TensorView<float> CyHzY,
    float invDx,
    float invDy
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hz);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hz.size(1), Hz.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float dEy_dx = (Ey[i + 1, j, k] - Ey[i, j, k]) * invDx;
    float dEx_dy = (Ex[i, j + 1, k] - Ex[i, j, k]) * invDy;
    float psiX = ByHzX[i] * PsiHzX[i, j, k] + CyHzX[i] * dEy_dx;
    float psiY = ByHzY[j] * PsiHzY[i, j, k] + CyHzY[j] * dEx_dy;
    PsiHzX[i, j, k] = psiX;
    PsiHzY[i, j, k] = psiY;

    float curlE = (dEy_dx * InvKappaHzX[i] + psiX) - (dEx_dy * InvKappaHzY[j] + psiY);
    Hz[i, j, k] = Hz[i, j, k] * HzDecay[i, j, k] - HzCurl[i, j, k] * curlE;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateMagneticFieldHxStandard3D(
    TensorView<float> Hx,
    TensorView<float> Ey,
    TensorView<float> Ez,
    TensorView<float> HxDecay,
    TensorView<float> HxCurl,
    float invDy,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hx);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hx.size(1), Hx.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float dEz_dy = (Ez[i, j + 1, k] - Ez[i, j, k]) * invDy;
    float dEy_dz = (Ey[i, j, k + 1] - Ey[i, j, k]) * invDz;
    float curlE = dEz_dy - dEy_dz;
    Hx[i, j, k] = Hx[i, j, k] * HxDecay[i, j, k] - HxCurl[i, j, k] * curlE;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateMagneticFieldHyStandard3D(
    TensorView<float> Hy,
    TensorView<float> Ex,
    TensorView<float> Ez,
    TensorView<float> HyDecay,
    TensorView<float> HyCurl,
    float invDx,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hy);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hy.size(1), Hy.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float dEx_dz = (Ex[i, j, k + 1] - Ex[i, j, k]) * invDz;
    float dEz_dx = (Ez[i + 1, j, k] - Ez[i, j, k]) * invDx;
    float curlE = dEx_dz - dEz_dx;
    Hy[i, j, k] = Hy[i, j, k] * HyDecay[i, j, k] - HyCurl[i, j, k] * curlE;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateMagneticFieldHzStandard3D(
    TensorView<float> Hz,
    TensorView<float> Ex,
    TensorView<float> Ey,
    TensorView<float> HzDecay,
    TensorView<float> HzCurl,
    float invDx,
    float invDy
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hz);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hz.size(1), Hz.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float dEy_dx = (Ey[i + 1, j, k] - Ey[i, j, k]) * invDx;
    float dEx_dy = (Ex[i, j + 1, k] - Ex[i, j, k]) * invDy;
    float curlE = dEy_dx - dEx_dy;
    Hz[i, j, k] = Hz[i, j, k] * HzDecay[i, j, k] - HzCurl[i, j, k] * curlE;
}

#endif
