#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "random.h"

#include "Payloads.h"
#include "Geometries.h"
#include "Light.h"
#include "Config.h"

#define IS_HEMISPEHRE 0
#define IS_COSINE 1
#define IS_BRDF 2

#define NEE_OFF 0
#define NEE_ON 1
#define NEE_MIS 2

#define BRDF_PHONG 0
#define BRDF_GGX 1

using namespace optix;

// Declare light buffers
rtBuffer<QuadLight> qlights;

// Declare variables
rtDeclareVariable(Payload, payload, rtPayload, );
rtDeclareVariable(rtObject, root, , );

rtDeclareVariable(int, lightSamples, , );
rtDeclareVariable(int, lightStratify, , );
rtDeclareVariable(int, nee, , );
rtDeclareVariable(int, importanceSampling, , );
rtDeclareVariable(int, russianRoulette, , );

rtDeclareVariable(int, brdf, , );
rtDeclareVariable(int, roughness, , );
rtDeclareVariable(int, gamma, , );

rtBuffer<Config> config; // Config

// Declare attibutes 
rtDeclareVariable(Attributes, attrib, attribute attrib, );


RT_PROGRAM void pathTracer() {
    MaterialValue mv = attrib.mv;
    Config cf = config[0];

	// ### SAMPLE ###
	float3 wi; 
	if (importanceSampling == IS_HEMISPEHRE) {
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
		wi = s.x * u + s.y * v + s.z * w;
	}
	else if (importanceSampling == IS_COSINE) {
		float theta = acos(sqrt(rnd(payload.seed)));
		float phi = 2 * M_PIf * rnd(payload.seed);
		float3 s = make_float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
		float3 w = normalize(attrib.normal);
		float3 a = make_float3(0, 1, 0);
		if (length(w - a) < cf.epsilon || length(w + a) < cf.epsilon) {//avoid a too close to w
			a = make_float3(1, 0, 0);
		}
		float3 u = normalize(cross(a, w));
		float3 v = cross(w, u);
		wi = s.x * u + s.y * v + s.z * w;
	}
	else if (importanceSampling == IS_BRDF) {
		// TODO
	}

	// ### BRDF ###
	float3 f; 
	if (brdf == BRDF_PHONG) {
		float3 rl = normalize(reflect(-attrib.wo, attrib.normal));
		f = mv.diffuse / M_PIf + mv.specular * (mv.shininess + 2) / (2 * M_PIf) * 
			pow(clamp(dot(rl, wi), 0.0f, 1.0f), mv.shininess);
	}
	else if (brdf == BRDF_GGX) {
		// TODO
	}

	// ### PDF ###
	float inv_pdf; 
	int N; 
	float3 throughput; 
	if (importanceSampling == IS_HEMISPEHRE) {
		inv_pdf = 2 * M_PIf;
		N = 1; 
		throughput = f * clamp(dot(attrib.normal, wi), 0.0f, 1.0f) * inv_pdf / N;
	}
	else if (importanceSampling == IS_COSINE) {
		inv_pdf = M_PIf;
		N = 1; 
		throughput = f * inv_pdf / N;
	}
	else if (importanceSampling == IS_BRDF) {
		// TODO
	}

	// ### NEE ###
    if (nee == NEE_ON) {

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
					// ### BRDF 2ND ###
					float3 f; 
					if (brdf == BRDF_PHONG) {
						f = mv.diffuse / M_PIf + mv.specular * (mv.shininess + 2) / (2 * M_PIf) * pow(clamp(dot(rl, wi), 0.0f, 1.0f), mv.shininess);
						float G = clamp(dot(sn, wi), 0.0f, 1.0f) * clamp(dot(ln, wi), 0.0f, 1.0f) / (lightDist * lightDist);
						f = f * G; 
					}
					else if (brdf == BRDF_GGX) {
						// TODO
					}
					tempResult += f;
				}
			}

			dlResult += qlights[i].color * A / lightSamples * tempResult;
		}

		// calculate radiance
		payload.radiance += payload.throughput * dlResult;
    }
    else if (nee == NEE_OFF) { 

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
	else if (nee == NEE_MIS) {
		// TODO
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


