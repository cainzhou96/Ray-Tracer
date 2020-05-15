#pragma once

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "Config.h"
#include "Geometries.h"
#include "Light.h"

struct Scene
{
    // Info about the output image
    std::string outputFilename;
    unsigned int width, height;

    Config config;

    std::string integratorName = "raytracer";

    std::vector<optix::float3> vertices;

    std::vector<Triangle> triangles;
    std::vector<Sphere> spheres;

    std::vector<DirectionalLight> dlights;
    std::vector<PointLight> plights;
    std::vector<QuadLight> qlights;

    int lightSamples = 1;
    bool lightStratify = false;

    int nee = 0; 
    int importanceSampling = 0; 
    bool russianRoulette = false;

    int brdf = 0; 
    float roughness = 1; 
    float gamma = 1; 

    Scene()
    {
        outputFilename = "raytrace.png";
        integratorName = "raytracer";
    }
};