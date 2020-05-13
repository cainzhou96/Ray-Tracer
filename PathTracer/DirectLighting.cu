#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "random.h"

#include "Payloads.h"
#include "Geometries.h"
#include "Light.h"
#include "Config.h"

using namespace optix;

// Declare light buffers
rtBuffer<QuadLight> qlights;

// Declare variables
rtDeclareVariable(Payload, payload, rtPayload, );
rtDeclareVariable(rtObject, root, , );

rtDeclareVariable(int, lightSamples, , );
rtDeclareVariable(int, lightStratify, , );

rtBuffer<Config> config; // Config

// Declare attibutes 
rtDeclareVariable(Attributes, attrib, attribute attrib, );

RT_PROGRAM void analytic()
{
    MaterialValue mv = attrib.mv;
    Config cf = config[0];

    float3 result = mv.ambient + mv.emission;
    float3 f = mv.diffuse / M_PIf;

    for (int i = 0; i < qlights.size(); i++) {
        float3 v[4];
        v[0] = qlights[i].a;
        v[1] = qlights[i].a + qlights[i].ab;
        v[2] = qlights[i].a + qlights[i].ab + qlights[i].ac;
        v[3] = qlights[i].a + qlights[i].ac;
        float3 r = attrib.intersection;
        float3 phi = make_float3(0, 0, 0);
        for (int k = 0; k < 4; k++) {
            float3 v_next = v[0];
            if (k < 3)
                v_next = v[k + 1];
            float theta = acos(dot(normalize(v[k] - r), normalize(v_next - r)));
            float3 gamma = normalize(cross((v[k] - r), (v_next - r)));
            phi += theta * gamma;
        }
        phi *= 0.5f;
        result += f * qlights[i].color * dot(phi, attrib.normal);
    }


    // Compute the final radiance
    payload.radiance = result * payload.throughput;
    payload.done = true;
    
}

RT_PROGRAM void monteCarlo()
{
    MaterialValue mv = attrib.mv;
    Config cf = config[0];

    float3 result = mv.ambient + mv.emission;
    for (int i = 0; i < qlights.size(); i++) {
        float3 tempResult = make_float3(0, 0, 0); 

        float A = length(cross(qlights[i].ab, qlights[i].ac)); 
        float3 hp = attrib.intersection;
        float3 sn = normalize(attrib.normal); 
        float3 ln = normalize(cross(qlights[i].ab, qlights[i].ac)); // ?
        float3 wo = normalize(attrib.wo); 
        float3 rl = normalize(reflect(-attrib.wo, attrib.normal)); // ?
        int stepNum = (int)sqrt((float)lightSamples); 
        for (int N = 0; N < lightSamples; N++) {
            float3 lp; 
            // randomize a light point
            if (lightStratify) {
                float3 abStep = qlights[i].ab / stepNum; 
                float3 acStep = qlights[i].ac / stepNum; 
                //rtPrintf("x : %f, y: %f, z: %f\n", abStep.x, abStep.y, abStep.z); 
                lp = qlights[i].a + (N % stepNum) * abStep + (N / stepNum) * acStep + rnd(payload.seed) * abStep + rnd(payload.seed) * acStep; 
                //rtPrintf("lp.x : %f, y: %f, z: %f\n", lp.x, lp.y, lp.z); 
            } else {
                lp = qlights[i].a + rnd(payload.seed) * qlights[i].ab + rnd(payload.seed) * qlights[i].ac; 
            }

            // check for shadow
            float3 lightDir = normalize(lp - hp);
            float lightDist = length(lp - hp);
            ShadowPayload shadowPayload;
            shadowPayload.isVisible = true;
            Ray shadowRay = make_Ray(hp, lightDir, 1, cf.epsilon, lightDist - cf.epsilon); // post @217
            rtTrace(root, shadowRay, shadowPayload);
            // If not in shadow
            if (shadowPayload.isVisible)
            {
                float3 wi = lightDir; 
                //rtPrintf("ln.x : %f, y: %f, z: %f\n", ln.x, ln.y, ln.z); 
                float3 f = mv.diffuse / M_PIf + mv.specular * (mv.shininess + 2) / (2 * M_PIf) * pow(clamp(dot(rl, wi), 0.0f, 1.0f), mv.shininess);
                float G = clamp(dot(sn, wi), 0.0f, 1.0f) * clamp(dot(ln, wi), 0.0f, 1.0f) / (lightDist * lightDist); 
                tempResult += f * G; 
            }
        }

        result += qlights[i].color * A / lightSamples * tempResult; 
    }

    // Compute the final radiance
    payload.radiance = result * payload.throughput;
    payload.done = true;

}