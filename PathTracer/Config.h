#pragma once

#include <optixu/optixu_math_namespace.h>

struct Config
{
    // Camera 
    optix::float3 w, u, v, eye; // w, u, v: orthonormal basis of camera; eye: eye location 
    optix::float2 hSize, tanHFov; // hSize: half size; tanHFov: tan of 0.5 * fov
    
    // Ray tracing 
    unsigned int maxDepth;
    float epsilon;
    int numFrames; 

    Config()
    {
        w = optix::make_float3(1.f, 0.f, 0.f);
        u = optix::make_float3(0.f, 1.f, 0.f);
        v = optix::make_float3(0.f, 0.f, 1.f);
        eye = optix::make_float3(0.f);
        hSize = optix::make_float2(100.f);
        tanHFov = optix::make_float2(tanf(0.25f * M_PIf));

        maxDepth = 5;
        epsilon = 0.0001f;
        numFrames = 1; 
    }
};