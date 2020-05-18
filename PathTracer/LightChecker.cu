#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>

#include "Payloads.h"
#include "Geometries.h"

using namespace optix;

rtDeclareVariable(LightPayload, payload, rtPayload, );
rtDeclareVariable(float3, backgroundColor, , );

rtDeclareVariable(Attributes, attrib, attribute attrib, );

RT_PROGRAM void closestHit()
{
    payload.hit = 1; 
    payload.intersection = attrib.intersection; 
    float3 n = normalize(attrib.normal); 
    float3 w = normalize(attrib.wo); 

    if (dot(n, w) > 0) {
        payload.emission = attrib.mv.emission; 
    }
    else {
        payload.emission = backgroundColor; 
    }
}