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

rtDeclareVariable(float, gamma, , );

rtBuffer<Config> config; // Config

// Declare attibutes 
rtDeclareVariable(Attributes, attrib, attribute attrib, );

float3 transformRay(float3 ray, float3 w); 
float3 getCosineSampleRay(); 
float3 getHemisphereSampleRay(); 
float3 getBRDFSampleRay(); 

float3 getPhongBRDF(float3 wi); 
float3 getGGXBRDF(float3 wi); 

float getCosinePDF(); 
float getHemispherePDF(); 
float getBRDFPDF(float3 wi); 
float getNeePDF(float3 wi); 

float3 getNEEDirectLighting(); 
float3 getBRDFDirectLighting(float3 wi); 

int isHittingLight(); 
float power(float base, int exp);
void printF3(float3 v); 

RT_PROGRAM void pathTracer() {
    MaterialValue mv = attrib.mv;
    Config cf = config[0];
	float3 wi;
	float3 throughput;

	// ### SAMPLE ###
	if (importanceSampling == IS_HEMISPEHRE) {
		wi = getHemisphereSampleRay();
	}
	else if (importanceSampling == IS_COSINE) {
		wi = getCosineSampleRay();
	}
	else if (importanceSampling == IS_BRDF) {
		wi = getBRDFSampleRay();
	}

	// ### BRDF ###
	float3 f; 
	if (mv.brdf == BRDF_PHONG) {
		f = getPhongBRDF(wi);
	}
	else if (mv.brdf == BRDF_GGX) {
		f = getGGXBRDF(wi);
	}

	// ### PDF ###
	float pdf;
	int N = 1;
	float3 n = normalize(attrib.normal); 
	if (importanceSampling == IS_HEMISPEHRE) {
		pdf = getHemispherePDF();
		throughput = f * clamp(dot(n, wi), 0.0f, 1.0f) / pdf / N;
	}
	else if (importanceSampling == IS_COSINE) {
		pdf = getCosinePDF();
		throughput = f / pdf / N;
	}
	else if (importanceSampling == IS_BRDF) {
		pdf = getBRDFPDF(wi);
		if (pdf <= 0) {
			throughput = make_float3(0, 0, 0); 
		}
		else {
			throughput = f * clamp(dot(n, wi), 0.0f, 1.0f) / pdf / N;
		}
	}

	// ### NEE ###
    if (nee == NEE_ON) {

		// check for hitting light
		int lightHit = isHittingLight(); 
		if (lightHit == 1) { // hit front 
			if (payload.depth == 0) {
				payload.radiance += payload.throughput * mv.emission;
			}
			payload.done = true; 
			return; 
		}
		else if (lightHit == 2) { // hit back
			payload.done = true;
			return;
		}

		// direct lighting result
		float3 dlResult = getNEEDirectLighting();

		// calculate radiance
		payload.radiance += payload.throughput * dlResult;
    }
    else if (nee == NEE_OFF) { 

		// check for hitting light
		int lightHit = isHittingLight();
		if (lightHit == 1) { // hit front 
			payload.radiance += payload.throughput * mv.emission;
			payload.done = true;
			return;
		}
		else if (lightHit == 2) { // hit back
			payload.done = true;
			return;
		}

        payload.radiance += payload.throughput * mv.emission;
	}
	else if (nee == NEE_MIS) {
		// check for hitting light
		int lightHit = isHittingLight();
		if (lightHit == 1) { // hit front 
			if (payload.depth == 0) {
				payload.radiance += payload.throughput * mv.emission;
			}
			payload.done = true;
			return;
		}
		else if (lightHit == 2) { // hit back
			payload.done = true;
			return;
		}

		int beta = 2; 
		float3 DLResult = mv.emission; 
		float3 curWi; 
		float3 curDLResult; 
		float3 curF; 
		float3 curThroughput; 
		float curBRDFPDF; 
		float curNEEPDF; 
		float curWeight; 

		// brdf
		curWi = getBRDFSampleRay();
		curDLResult = getBRDFDirectLighting(curWi);
		curBRDFPDF = getBRDFPDF(curWi); 
		curNEEPDF = getNeePDF(curWi); 
		n = normalize(attrib.normal); 
		if (curBRDFPDF <= 0) {
			curThroughput = make_float3(0); 
		}
		else {
			curWeight = power(curBRDFPDF, beta) / (power(curBRDFPDF, beta) + power(curNEEPDF, beta)); 
			if (mv.brdf == BRDF_PHONG) {
				curF = getPhongBRDF(curWi); 
			}
			else if (mv.brdf == BRDF_GGX) {
				curF = getGGXBRDF(curWi); 
			}
			curThroughput = curWeight * curF * clamp(dot(n, curWi), 0.0f, 1.0f) / curBRDFPDF; 
		}
		DLResult += curThroughput * curDLResult; 

		// nee
		curDLResult = make_float3(0);
		for (int i = 0; i < qlights.size(); i++) {
			float A = length(cross(qlights[i].ab, qlights[i].ac));
			float3 hp = attrib.intersection;
			float3 sn = normalize(attrib.normal);
			float3 ln = -normalize(cross(qlights[i].ab, qlights[i].ac));
			float3 wo = normalize(attrib.wo);
			float3 rl = normalize(reflect(-wo, sn));
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
				Ray shadowRay = make_Ray(hp, lightDir, 1, config[0].epsilon, lightDist - config[0].epsilon);
				rtTrace(root, shadowRay, shadowPayload);
				// If not in shadow
				if (shadowPayload.isVisible)
				{
					curDLResult = qlights[i].color; 
					curWi = lightDir; 
					curBRDFPDF = getBRDFPDF(curWi); 
					curNEEPDF = getNeePDF(curWi); 
					if (curNEEPDF == 0) {
						curThroughput = make_float3(0); // hack it for now
					}
					else {
						if (curBRDFPDF <= 0) {
							curBRDFPDF = 0; 
						}
						curWeight = power(curNEEPDF, beta) / (power(curBRDFPDF, beta) + power(curNEEPDF, beta)); 
						if (mv.brdf == BRDF_PHONG) {
							curF = getPhongBRDF(curWi); 
						}
						else if (mv.brdf == BRDF_GGX) {
							curF = getGGXBRDF(curWi); 
						}
						float G = clamp(dot(sn, curWi), 0.0f, 1.0f) * clamp(dot((-ln), curWi), 0.0f, 1.0f) / (lightDist * lightDist);
						curThroughput = curWeight * curF * G * A / lightSamples; 
						DLResult += curThroughput * curDLResult; 
					}
				}
			}
		}
		payload.radiance += payload.throughput * DLResult;
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

float3 transformRay(float3 ray, float3 w) {
	float3 a = make_float3(0, 1, 0);
	if (length(w - a) < config[0].epsilon || length(w + a) < config[0].epsilon) {//avoid a too close to w
		a = make_float3(1, 0, 0);
	}
	float3 u = normalize(cross(a, w));
	float3 v = cross(w, u);
	return ray.x * u + ray.y * v + ray.z * w;
}

float3 getHemisphereSampleRay() {
	float3 wi; 
	float theta = acosf(rnd(payload.seed));
	float phi = 2 * M_PIf * rnd(payload.seed);
	float3 s = make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
	float3 w = normalize(attrib.normal);
	wi = transformRay(s, w);
	wi = normalize(wi); 
	return wi; 
}

float3 getCosineSampleRay() {
	float3 wi; 
	float theta = acosf(sqrt(rnd(payload.seed)));
	float phi = 2 * M_PIf * rnd(payload.seed);
	float3 s = make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
	float3 w = normalize(attrib.normal);
	wi = transformRay(s, w); 
	wi = normalize(wi);
	return wi; 
}

float3 getBRDFSampleRay() {
	MaterialValue mv = attrib.mv; 
	float3 wi; 
	float3 wo = normalize(attrib.wo); 
	float3 n = normalize(attrib.normal); 
	float ks = (mv.specular.x + mv.specular.y + mv.specular.z) / 3.0f;
	float kd = (mv.diffuse.x + mv.diffuse.y + mv.diffuse.z) / 3.0f;
	if (mv.brdf == BRDF_PHONG) {
		float3 rl = normalize(-reflect(wo, n));
		float t = ks / (ks + kd);

		float phi = 2 * M_PIf * rnd(payload.seed);
		float theta = 0;
		float3 s, w;
		if (rnd(payload.seed) <= t) { //specular
			theta = acosf(power(rnd(payload.seed), 1 / (mv.shininess + 1)));
			s = make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
			w = rl;
		}
		else { // diffuse
			theta = acosf(sqrtf(rnd(payload.seed)));
			s = make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
			w = normalize(n);
		}
		wi = transformRay(s, w); 
	}
	else if (mv.brdf == BRDF_GGX) {
		float t = fmaxf(0.25f, ks / (ks + kd));
		float phi = 2 * M_PIf * rnd(payload.seed);

		if (rnd(payload.seed) <= t) { // specular
			float rand = rnd(payload.seed);
			float theta = atanf(mv.roughness * sqrtf(rand) / sqrtf(1 - rand));
			float3 h = make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
			float3 w = n;
			h = transformRay(h, w);
			wi = reflect(-wo, h);
		}
		else { // diffuse
			float theta = acosf(sqrtf(rnd(payload.seed)));
			float3 s = make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
			float3 w = n;
			wi = transformRay(s, w);
		}
	}
	wi = normalize(wi);
	return wi; 
}

float3 getPhongBRDF(float3 wi) {
	MaterialValue mv = attrib.mv; 
	float3 wo = normalize(attrib.wo); 
	float3 n = normalize(attrib.normal); 
	float3 rl = normalize(reflect(-wo, n));
	float3 f; 
	f = mv.diffuse / M_PIf + mv.specular * (mv.shininess + 2) / (2 * M_PIf) *
		power(dot(rl, wi), mv.shininess);
	return f; 
}

float3 getGGXBRDF(float3 wi) {
	MaterialValue mv = attrib.mv; 
	float3 wo = normalize(attrib.wo);
	float3 n = normalize(attrib.normal);
	float3 h = normalize(wi + wo);
	float3 f_ggx;
	float3 f; 
	if (dot(wi, n) <= 0 || dot(wo, n) <= 0) {
		f_ggx = make_float3(0, 0, 0);
	}
	else {
		float alpha_square = mv.roughness * mv.roughness;
		float theta_h; 
		if (dot(h, n) < 1) {
			theta_h = acosf(dot(h, n));
		}
		else {
			theta_h = 0; 
		}
		float D = alpha_square / (M_PIf * power(cosf(theta_h), 4) *
			power((alpha_square + tanf(theta_h) * tanf(theta_h)), 2));

		float theta_wi = acosf(dot(wi, n));
		float G1_wi = 2.0f / (1 + sqrtf(1 + alpha_square * tanf(theta_wi) * tanf(theta_wi)));
		float theta_wo = acosf(dot(wo, n));
		float G1_wo = 2.0f / (1 + sqrtf(1 + alpha_square * tanf(theta_wo) * tanf(theta_wo)));
		float G = G1_wi * G1_wo;

		float3 F = mv.specular + (make_float3(1.0f, 1.0f, 1.0f) - mv.specular) * power((1 - dot(wi, h)), 5);
		f_ggx = F * G * D / (4 * dot(wi, n) * dot(wo, n));
	}
	f = mv.diffuse / M_PIf + f_ggx;
	return f;
}

float getHemispherePDF() {
	return 1 / (2 * M_PIf);
}

float getCosinePDF() {
	return 1 / M_PIf; 
}

float getBRDFPDF(float3 wi) {
	MaterialValue mv = attrib.mv; 
	float3 wo = normalize(attrib.wo);
	float3 n = normalize(attrib.normal); 
	float3 rl = normalize(reflect(-wo, n));
	float ks = (mv.specular.x + mv.specular.y + mv.specular.z) / 3.0f;
	float kd = (mv.diffuse.x + mv.diffuse.y + mv.diffuse.z) / 3.0f;
	float pdf; 
	if (mv.brdf == BRDF_PHONG) {
		float t = ks / (ks + kd);
		if (isnan(t))
			t = 0;
		pdf = (1 - t) * clamp(dot(n, wi), 0.0f, 1.0f) / M_PIf +
			t * (mv.shininess + 1) / (2 * M_PIf) * power(clamp(dot(rl, wi), 0.0f, 1.0f), mv.shininess);
	}
	else if (mv.brdf == BRDF_GGX) {
		float t = fmaxf(0.25f, ks / (ks + kd));
		float3 h = normalize(wi + wo);
		float alpha_square = mv.roughness * mv.roughness;
		float theta_h; 
		if (dot(h, n) < 1) {
			theta_h = acosf(dot(h, n));
		}
		else {
			theta_h = 0; 
		}
		float D = alpha_square / (M_PIf * power(cosf(theta_h), 4) *
			power((alpha_square + tanf(theta_h) * tanf(theta_h)), 2));
		pdf = (1 - t) * clamp(dot(n, wi), 0.0f, 1.0f) / M_PIf + t * D * dot(n, h) / (4 * dot(h, wi));
	}
	return pdf; 
}


float getNeePDF(float3 wi) {
	if (qlights.size() == 0)
		return 0;

	float pdf_nee = 0;
	// check for hitting light
	for (int i = 0; i < qlights.size(); i++) {
		QuadLight q = qlights[i];
		float3 ln = -normalize(cross(qlights[i].ab, qlights[i].ac));
		float t = dot(qlights[i].a - attrib.intersection, ln) / dot(wi, ln);
		if (t > 0) {
			float3 hp = attrib.intersection + wi * t;
			float u = dot(hp - q.a, q.ab);
			float v = dot(hp - q.a, q.ac);
			// hit quad light (MAYBE INCORRECT)
			float3 ab = qlights[i].ab + config[0].epsilon * normalize(qlights[i].ab);
			float3 ac = qlights[i].ac + config[0].epsilon * normalize(qlights[i].ac);
			if (u >= 0 && u <= dot(ab, ab) && v >= 0 && v <= dot(ac, ac)) {
				float A = length(cross(qlights[i].ab, qlights[i].ac));
				float R = fabsf(t);
				pdf_nee += R * R / (A * fabsf(dot(ln, wi)));
			}
		}
	}
	pdf_nee = pdf_nee / qlights.size();
	return pdf_nee;

	/*
	if (qlights.size() == 0)
		return 0;

	float pdf_nee = 0;
	LightPayload lightPayload;
	lightPayload.hit = 0;
	lightPayload.emission = make_float3(0);
	Ray ray = make_Ray(attrib.intersection, wi, 2, config[0].epsilon, RT_DEFAULT_MAX);
	rtTrace(root, ray, lightPayload);
	if (!lightPayload.hit) {
		return 0; 
	}
	for (int i = 0; i < qlights.size(); i++) {
		float cur_pdf = 0; 
		float3 ln = -normalize(cross(qlights[i].ab, qlights[i].ac));
		float t = -(dot(qlights[i].a, ln) - dot(lightPayload.intersection, ln));
		if (t < config[0].epsilon && t > -config[0].epsilon) { // hitting a light
			float A = length(cross(qlights[i].ab, qlights[i].ac));
			float R = length(lightPayload.intersection - attrib.intersection);
			cur_pdf = R * R / A / abs(dot(ln, wi)); 
		}
		pdf_nee += cur_pdf; 
	}

	pdf_nee = pdf_nee / qlights.size();
	return pdf_nee;
	*/
}

float3 getNEEDirectLighting() {
	MaterialValue mv = attrib.mv; 
	float3 dlResult = mv.emission;
	for (int i = 0; i < qlights.size(); i++) {
		float3 tempResult = make_float3(0, 0, 0);

		float A = length(cross(qlights[i].ab, qlights[i].ac));
		float3 hp = attrib.intersection;
		float3 sn = normalize(attrib.normal);
		float3 ln = -normalize(cross(qlights[i].ab, qlights[i].ac));
		float3 wo = normalize(attrib.wo);
		float3 rl = normalize(reflect(-wo, sn));
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
			Ray shadowRay = make_Ray(hp, lightDir, 1, config[0].epsilon, lightDist - config[0].epsilon);
			rtTrace(root, shadowRay, shadowPayload);
			// If not in shadow
			if (shadowPayload.isVisible)
			{
				// ### BRDF 2ND ###
				float3 f;
				float G = clamp(dot(sn, lightDir), 0.0f, 1.0f) * clamp(dot((-ln), lightDir), 0.0f, 1.0f) / (lightDist * lightDist);
				if (mv.brdf == BRDF_PHONG) {
					f = getPhongBRDF(lightDir);
					f = f * G;
				}
				else if (mv.brdf == BRDF_GGX) {
					f = getGGXBRDF(lightDir);
					f = f * G;
				}
				tempResult += f;
			}
		}

		dlResult += qlights[i].color * A / lightSamples * tempResult;
	}
	return dlResult; 
}

float3 getBRDFDirectLighting(float3 wi) {
	MaterialValue mv = attrib.mv;
	LightPayload lightPayload;
	lightPayload.hit = 0; 
	lightPayload.emission = make_float3(0);
	Ray ray = make_Ray(attrib.intersection, wi, 2, config[0].epsilon, RT_DEFAULT_MAX);
	rtTrace(root, ray, lightPayload);
	return lightPayload.emission; 
}

int isHittingLight() { // return 0: not hitting light, 1: hitting front, 2: hitting back
	for (int i = 0; i < qlights.size(); i++) {
		float3 ln = -normalize(cross(qlights[i].ab, qlights[i].ac));
		float t = -(dot(qlights[i].a, ln) - dot(attrib.intersection, ln));
		if (t < config[0].epsilon && t > -config[0].epsilon) { // hitting a light
			if (dot(ln, normalize(attrib.wo)) > 0) {
				return 1; 
			}
			else {
				return 2;
			}
		}
	}
	return 0; 
}

float power(float base, int exp) {
	if (exp == 0)
		return 1.0f;
	float res = base;
	for (int i = 1; i < exp; i++) {
		res *= base;
	}
	return res;
}

void printF3(float3 v) {
	rtPrintf("%f, %f, %f\n", v.x, v.y, v.z); 
}

