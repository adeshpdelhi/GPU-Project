#ifdef OBJECT_H
#define OBJECT_COUNT 1500 //Do NOT go beyond 1500
#define MAX_MAPPINGS 100000000
#define BLOCK_DIM 1024

#include "glm/glm.hpp"
#include <bits/stdc++.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <climits>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#define MAX(a,b) ((a > b) ? a : b)

class object{
public:
    int n_vertices;
    float4 speed;
    float4 centroid;
    float4 initial_location;
    float rotation_matrix[4][4];


}objects[OBJECT_COUNT];

bool loadOBJ(
    const char * path, 
    std::vector<glm::vec3> & out_vertices, 
    std::vector <unsigned int> & mappings
);
float getMaximumBoundingBox(std::vector <std::vector <glm::vec3> > );

GLuint vbo;
GLuint ibo;

struct cudaGraphicsResource *cuda_vbo_resource;


float boundingBoxLength;

std::vector<glm::vec3> vertices;
std::vector<unsigned int> mappings;


#endif