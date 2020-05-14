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
rtDeclareVariable(float, roughness, , );
rtDeclareVariable(float, gamma, , );

rtBuffer<Config> config; // Config

// Declare attibutes 
rtDeclareVariable(Attributes, attrib, attribute attrib, );

float3 transformRay(float3 ray, float3 w, float epsilon); 
float3 getCosineSampleRay(float epsilon); 
float3 getHemisphereSampleRay(float epsilon); 
float3 getBRDFSampleRay(Attributes attrib, float epsilon); 

float3 getPhongBRDF(Attributes attrib, float3 wi); 
float3 getGGXBRDF(Attributes attrib, float3 wi); 
float3 getGGXThroughput(Attributes attrib, float3& wi);

float getCosinePDF(); 
float getHemispherePDF(); 
float getBRDFPDF(Attributes attrib, float3 wi); 

RT_PROGRAM void pathTracer() {
    MaterialValue mv = attrib.mv;
    Config cf = config[0];
	float3 wi;
	float3 throughput;

	if (brdf == BRDF_GGX) {
		throughput = getGGXThroughput(attrib, wi);
	}
	else {

		// ### SAMPLE ###
		if (importanceSampling == IS_HEMISPEHRE) {
			wi = getHemisphereSampleRay(cf.epsilon);
		}
		else if (importanceSampling == IS_COSINE) {
			wi = getCosineSampleRay(cf.epsilon);
		}
		else if (importanceSampling == IS_BRDF) {
			wi = getBRDFSampleRay(attrib, cf.epsilon);
		}

		// ### BRDF ###
		float3 f = getPhongBRDF(attrib, wi);

		// ### PDF ###
		float pdf;
		int N = 1;
		if (importanceSampling == IS_HEMISPEHRE) {
			pdf = getHemispherePDF();
			throughput = f * clamp(dot(attrib.normal, wi), 0.0f, 1.0f) / pdf / N;
		}
		else if (importanceSampling == IS_COSINE) {
			pdf = getCosinePDF();
			throughput = f / pdf / N;
		}
		else if (importanceSampling == IS_BRDF) {
			pdf = getBRDFPDF(attrib, wi);
			throughput = f * clamp(dot(attrib.normal, wi), 0.0f, 1.0f) / pdf / N;
		}
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
						f = getGGXBRDF(attrib, wi);
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

float3 transformRay(float3 ray, float3 w, float epsilon) {
	float3 a = make_float3(0, 1, 0);
	if (length(w - a) < epsilon || length(w + a) < epsilon) {//avoid a too close to w
		a = make_float3(1, 0, 0);
	}
	float3 u = normalize(cross(a, w));
	float3 v = cross(w, u);
	return ray.x * u + ray.y * v + ray.z * w;
}

float3 getHemisphereSampleRay(float epsilon) {
	float3 wi; 
	float theta = acosf(rnd(payload.seed));
	float phi = 2 * M_PIf * rnd(payload.seed);
	float3 s = make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
	float3 w = normalize(attrib.normal);
	wi = transformRay(s, w, epsilon);
	return wi; 
}

float3 getCosineSampleRay(float epsilon) {
	float3 wi; 
	float theta = acosf(sqrt(rnd(payload.seed)));
	float phi = 2 * M_PIf * rnd(payload.seed);
	float3 s = make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
	float3 w = normalize(attrib.normal);
	wi = transformRay(s, w, epsilon); 
	return wi; 
}

float3 getBRDFSampleRay(Attributes attrib, float epsilon) {
	MaterialValue mv = attrib.mv; 
	float3 wi; 
	float3 rl = normalize(reflect(-attrib.wo, attrib.normal));
	float ks = (mv.specular.x + mv.specular.y + mv.specular.z) / 3.0f;
	float kd = (mv.diffuse.x + mv.diffuse.y + mv.diffuse.z) / 3.0f;
	float t = ks / (ks + kd);

	float phi = 2 * M_PIf * rnd(payload.seed);
	float theta = 0;
	float3 s, w;
	if (rnd(payload.seed) <= t) { //specular
		theta = acosf(powf(rnd(payload.seed), 1 / (mv.shininess + 1)));
		s = make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
		w = rl;
	}
	else { // diffuse
		theta = acosf(sqrt(rnd(payload.seed)));
		s = make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
		w = normalize(attrib.normal);
	}
	wi = transformRay(s, w, epsilon); 
	return wi; 
}

float3 getPhongBRDF(Attributes attrib, float3 wi) {
	MaterialValue mv = attrib.mv; 
	float3 f; 
	float3 rl = normalize(reflect(-attrib.wo, attrib.normal));
	f = mv.diffuse / M_PIf + mv.specular * (mv.shininess + 2) / (2 * M_PIf) *
		pow(clamp(dot(rl, wi), 0.0f, 1.0f), mv.shininess);
	return f; 
}

float3 getGGXBRDF(Attributes attrib, float3 wi) {
	MaterialValue mv = attrib.mv; 
	float3 n = attrib.normal;
	float3 h = normalize(wi + attrib.wo);
	float alpha_cube = roughness * roughness;
	float theta_h = acosf(dot(h, n));
	float D = alpha_cube / (M_PIf * powf(cosf(theta_h), 4) *
		powf((alpha_cube + tanf(theta_h) * tanf(theta_h)), 2));
	
	float theta_wi = acosf(dot(wi, n));
	float G1_wi = dot(wi, n) > 0 ?
		2.0f / (1 + sqrtf(1 + alpha_cube * tanf(theta_wi) * tanf(theta_wi))) : 0;
	float theta_wo = acosf(dot(attrib.wo, n));
	float G1_wo = dot(attrib.wo, n) > 0 ?
		2.0f / (1 + sqrtf(1 + alpha_cube * tanf(theta_wo) * tanf(theta_wo))) : 0;
	float G = G1_wi * G1_wo;

	float3 F = mv.specular + (1 - mv.specular) * powf((1 - dot(wi, h)), 5);
	float3 f_ggx = F * G * D / (4 * dot(wi, n) * dot(attrib.wo, n));
	float3 f = mv.diffuse / M_PIf + f_ggx;

	return f;
}

float getHemispherePDF() {
	return 1 / (2 * M_PIf);
}

float getCosinePDF() {
	return 1 / M_PIf; 
}

float getBRDFPDF(Attributes attrib, float3 wi) {
	MaterialValue mv = attrib.mv; 
	float3 rl = normalize(reflect(-attrib.wo, attrib.normal));
	float ks = (mv.specular.x + mv.specular.y + mv.specular.z) / 3.0f;
	float kd = (mv.diffuse.x + mv.diffuse.y + mv.diffuse.z) / 3.0f;
	float t = ks / (ks + kd);
	float pdf = (1 - t) * clamp(dot(attrib.normal, wi), 0.0f, 1.0f) / M_PIf +
		t * (mv.shininess + 1) / (2 * M_PIf) * pow(clamp(dot(rl, wi), 0.0f, 1.0f), mv.shininess);
	return pdf; 
}


float3 getGGXThroughput(Attributes attrib, float3& wi) {
	MaterialValue mv = attrib.mv;

	float ks = (mv.specular.x + mv.specular.y + mv.specular.z) / 3.0f;
	float kd = (mv.diffuse.x + mv.diffuse.y + mv.diffuse.z) / 3.0f;
	float t = fmaxf(0.25f, ks / (ks + kd));
	float3 n = attrib.normal;

	// sample
	if (rnd(payload.seed) <= t) { // specular
		float phi = 2 * M_PIf * rnd(payload.seed);
		float rand = rnd(payload.seed);
		float theta = atanf(roughness * sqrtf(rand) / sqrtf(1 - rand));
		float3 h = make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
		h = transformRay(h, n, config[0].epsilon);
		wi = reflect(-attrib.wo, h);
	}
	else { // diffuse
		float phi = 2 * M_PIf * rnd(payload.seed);
		float theta = acosf(sqrtf(rnd(payload.seed)));
		float3 s = make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
		wi = transformRay(s, n, config[0].epsilon);
	}

	// BRDF
	float3 h = normalize(wi + attrib.wo);
	float alpha_cube = roughness * roughness;
	float theta_h = acosf(dot(h, n));
	float D = alpha_cube / (M_PIf * powf(cosf(theta_h), 4) *
		powf((alpha_cube + tanf(theta_h) * tanf(theta_h)), 2));

	float theta_wi = acosf(dot(wi, n));
	float G1_wi = dot(wi, n) > 0 ?
		2.0f / (1 + sqrtf(1 + alpha_cube * tanf(theta_wi) * tanf(theta_wi))) : 0;
	float theta_wo = acosf(dot(attrib.wo, n));
	float G1_wo = dot(attrib.wo, n) > 0 ?
		2.0f / (1 + sqrtf(1 + alpha_cube * tanf(theta_wo) * tanf(theta_wo))) : 0;
	float G = G1_wi * G1_wo;

	float3 F = mv.specular + (1 - mv.specular) * powf((1 - dot(wi, h)), 5);
	float3 f_ggx = F * G * D / (4 * dot(wi, n) * dot(attrib.wo, n));
	float3 f = mv.diffuse / M_PIf + f_ggx;

	//PDF
	float pdf = (1 - t) * dot(n, wi) / M_PIf + t * D * dot(n, h) / (4 * dot(h, wi));

	float3 throughput = f * clamp(dot(attrib.normal, wi), 0.0f, 1.0f) / pdf;
	return throughput;
}