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
rtDeclareVariable(int, nee, , );
rtDeclareVariable(int, russianRoulette, , );

rtBuffer<Config> config; // Config

// Declare attibutes 
rtDeclareVariable(Attributes, attrib, attribute attrib, );


RT_PROGRAM void pathTracer() {
    MaterialValue mv = attrib.mv;
    Config cf = config[0];
    //float3 result = mv.emission;

    // find wi
    float theta = acos(rnd(payload.seed));
    float phi = 2 * M_PIf * rnd(payload.seed);
    float3 s = make_float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
    float3 w = normalize(attrib.normal);
    float3 a = make_float3(0, 1, 0);
    if (length(w - a) < cf.epsilon || length(w + a) < cf.epsilon) {//avoid a too close to w
        a = make_float3(1, 0, 0);
    }
    float3 u = normalize(cross(a, w));
    float3 v = cross(w, u);
    float3 wi = s.x * u + s.y * v + s.z * w;

    float3 rl = normalize(reflect(-attrib.wo, attrib.normal));
    float3 f = mv.diffuse / M_PIf + mv.specular * (mv.shininess + 2) / (2 * M_PIf) * 
        pow(clamp(dot(rl, wi), 0.0f, 1.0f), mv.shininess);
    float inv_pdf = 2 * M_PIf;
	int N = 1; 
    float3 throughput = f * clamp(dot(attrib.normal, wi), 0.0f, 1.0f) * inv_pdf / N;

    if (nee) {

		// check for hitting light
		for (int i = 0; i < qlights.size(); i++) {
			float3 ln = normalize(cross(qlights[i].ab, qlights[i].ac));
			float t = - (dot(qlights[i].a, ln) - dot(attrib.intersection, ln));
			if (t < cf.epsilon && t > -cf.epsilon) { // hitting a light
				if (payload.depth == 0) {
					payload.radiance += mv.emission;
				}
				payload.done = true; 
				return; 
			}
		}

		// direct lighting
		float3 dlResult = mv.emission;
		for (int i = 0; i < qlights.size(); i++) {
			float3 tempResult = make_float3(0, 0, 0);

			float A = length(cross(qlights[i].ab, qlights[i].ac));
			float3 hp = attrib.intersection;
			float3 sn = normalize(attrib.normal);
			float3 ln = normalize(cross(qlights[i].ab, qlights[i].ac));
			float3 wo = normalize(attrib.wo);
			float3 rl = normalize(reflect(-attrib.wo, attrib.normal));
			int stepNum = (int)sqrt((float)lightSamples);
			for (int ls = 0; ls < lightSamples; ls++) {
				float3 lp;
				// randomize a light point
				if (lightStratify) {
					float3 abStep = qlights[i].ab / stepNum;
					float3 acStep = qlights[i].ac / stepNum;
					lp = qlights[i].a + (ls % stepNum) * abStep + (ls / stepNum) * acStep + rnd(payload.seed) * abStep + rnd(payload.seed) * acStep;
				}
				else {
					lp = qlights[i].a + rnd(payload.seed) * qlights[i].ab + rnd(payload.seed) * qlights[i].ac;
				}

				// check for shadow
				float3 lightDir = normalize(lp - hp);
				float lightDist = length(lp - hp);
				ShadowPayload shadowPayload;
				shadowPayload.isVisible = true;
				Ray shadowRay = make_Ray(hp, lightDir, 1, cf.epsilon, lightDist - cf.epsilon);
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

			dlResult += qlights[i].color * A / lightSamples * tempResult;
		}

		// calculate radiance
		payload.radiance += payload.throughput * dlResult;
    }
    else { // not nee

		// check for hitting light
		for (int i = 0; i < qlights.size(); i++) {
			float3 ln = normalize(cross(qlights[i].ab, qlights[i].ac));
			float t = -(dot(qlights[i].a, ln) - dot(attrib.intersection, ln));
			if (t < cf.epsilon && t > -cf.epsilon) { // hitting a light
				payload.radiance += payload.throughput * mv.emission;
				payload.done = true;
				return;
			}
		}

        payload.radiance += payload.throughput * mv.emission;
    }

	// calculate Russian Roulette
	if (russianRoulette) {
		float q = 1 - fminf(fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)), 1.0f);
		if (rnd(payload.seed) <= q) { //terminate
			payload.done = true;
			return;
		}
		else {
			float boost = 1.0f / (1.0f - q);
			throughput *= boost;
		}
	}

    
    // for recursion
    payload.origin = attrib.intersection;
    payload.dir = wi; 
    //add throughput for next iteration
    payload.throughput = payload.throughput * throughput;
    payload.depth++; 
}


