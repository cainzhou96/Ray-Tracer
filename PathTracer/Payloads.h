#pragma once

#include <optixu/optixu_math_namespace.h>
#include "Geometries.h"

/**
 * Structures describing different payloads should be defined here.
 */

struct Payload
{
    optix::float3 radiance, throughput, origin, dir;
    unsigned int depth, seed;
    bool done;
};


struct ShadowPayload
{
    int isVisible;
};