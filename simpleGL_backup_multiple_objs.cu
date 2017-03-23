////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/

#define OBJECT_COUNT 4
#define BLOCK_DIM 1024
// includes, system
// #include <GL/glew.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#include <thrust/device_vector.h>
#include <bits/stdc++.h>


#define GLM_FORCE_CUDA
#include "glm/glm.hpp"


#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width    = 256;
const unsigned int mesh_height   = 256;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *sSDKsample = "simpleGL (VBO)";

struct object{
    int n_vertices;
    float4 speed;
}objects[OBJECT_COUNT];



std::vector<glm::vec3> vertices;
std::vector<glm::vec2> uvs;
std::vector<glm::vec3> normals;

bool loadOBJ(
    const char * path, 
    std::vector<glm::vec3> & out_vertices, 
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals
);

__device__ int getObjectId(int index, struct object* d_objects){
    int sum = 0;
    for (int i = 0; i < OBJECT_COUNT; ++i)
    {
        sum = sum + d_objects[i].n_vertices;
        if(index < sum)
        {
            // if(i==0)
            // printf("Sent %d\n", i);
            return i;
        }

    }
    // printf("getObjectId: Object Id not found. %d %d %d %d Sending -1\n",index, sum, d_objects[0].n_vertices, d_objects[1].n_vertices);
    return -1;
}

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4 *pos, struct object* d_objects, float time)
{
    // unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    // unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // printf("here\n");

    int object_id = getObjectId(idx, d_objects);
    if(object_id == -1)
        return;
    // printf("now here\n");
    
    float4 speed = d_objects[object_id].speed;
    // printf("%d: %f %f %f\n", object_id,speed.x, speed.y, speed.z);
    // printf("%d: %f %f %f\n", idx, pos[idx].x, pos[idx].y, pos[idx].z);
    pos[idx] = make_float4(pos[idx].x + speed.x*time, pos[idx].z + speed.z*time, 
        pos[idx].y+speed.y*time, 1.0f);
    // printf("%d: %f %f %f\n", idx, pos[idx].x, pos[idx].y, pos[idx].z);

    // pos[y*width+x] = make_float4(u, w, v, 1.0f);
}



void launch_kernel(float4 *pos, struct object* objects, float time)
{
    // execute the kernel
    dim3 grid(ceil((float)vertices.size()/BLOCK_DIM),1);
    dim3 block(BLOCK_DIM,1);
    simple_vbo_kernel<<< grid, block>>>(pos, objects, time);
}

bool checkHW(char *name, const char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int findGraphicsGPU(char *name)
{
    int nGraphicsGPU = 0;
    int deviceCount = 0;
    bool bFoundGraphics = false;
    char firstGraphicsName[256], temp[256];

    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("> FAILED %s sample finished, exiting...\n", sSDKsample);
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else
    {
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
        printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

        if (bGraphics)
        {
            if (!bFoundGraphics)
            {
                strcpy(firstGraphicsName, temp);
            }

            nGraphicsGPU++;
        }
    }

    if (nGraphicsGPU)
    {
        strcpy(name, firstGraphicsName);
    }
    else
    {
        strcpy(name, "this hardware");
    }

    return nGraphicsGPU;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    char *ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("%s starting...\n", sSDKsample);

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
            getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
        }
    }


    runTest(argc, argv, ref_file);

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // command line mode only
    if (ref_file != NULL)
    {
        // This will pick the best possible CUDA capable device
        int devID = findCudaDevice(argc, (const char **)argv);

        // create VBO
        checkCudaErrors(cudaMalloc((void **)&d_vbo_buffer, mesh_width*mesh_height*4*sizeof(float)));

        // run the cuda part
        runAutoTest(devID, argv, ref_file);

        // check result of Cuda step
        checkResultCuda(argc, argv, vbo);

        cudaFree(d_vbo_buffer);
        d_vbo_buffer = NULL;
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        if (false == initGL(&argc, argv))
        {
            return false;
        }

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
            if (gpuGLDeviceInit(argc, (const char **)argv) == -1)
            {
                return false;
            }
        }
        else
        {
            cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
        }

        // register callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

        // create VBO
        createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

        // run the cuda part
        runCuda(&cuda_vbo_resource);

        // start rendering mainloop
        glutMainLoop();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
    // printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    // execute the kernel
    //    dim3 block(8, 8, 1);
    //    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    //    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);
    
    // printf("vertices size: %d\n",vertices.size());

    float4 host_pos[vertices.size()];
    for (int i = 0; i<vertices.size(); ++i)
    {
        // printf("i = %d\n",i);
        // printf("vertices[i].x = %f\n",vertices[i].x);
        // printf("pos[i].x = %f\n", host_pos[i].x);
        host_pos[i] = make_float4(vertices[i].x,vertices[i].y, vertices[i].z,1.0f);
    }
    cudaMemcpy(dptr, host_pos, sizeof(host_pos), cudaMemcpyHostToDevice);
    struct object* d_objects;
    cudaMalloc(&d_objects, OBJECT_COUNT * sizeof(object));
    cudaMemcpy(d_objects, objects, sizeof(objects), cudaMemcpyHostToDevice);
    launch_kernel(dptr, d_objects, g_fAnim);
    cudaFree(d_objects);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
    printf("sdkDumpBin: <%s>\n", filename);
    FILE *fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char **argv, char *ref_file)
{
    char *reference_file = NULL;
    void *imageData = malloc(mesh_width*mesh_height*sizeof(float));

    // execute the kernel
    launch_kernel((float4 *)d_vbo_buffer, objects, g_fAnim);

    cudaDeviceSynchronize();
    getLastCudaError("launch_kernel failed");

    checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, mesh_width*mesh_height*sizeof(float), cudaMemcpyDeviceToHost));

    sdkDumpBin2(imageData, mesh_width*mesh_height*sizeof(float), "simpleGL.bin");
    reference_file = sdkFindFilePath(ref_file, argv[0]);

    if (reference_file &&
        !sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
                                mesh_width*mesh_height*sizeof(float),
                                MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
    {
        g_TotalErrors++;
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    // unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);


    bool res = loadOBJ("cone.obj", vertices, uvs, normals);
    if(res == false)
        exit(1);
    objects[0].n_vertices = vertices.size();
    objects[0].speed = make_float4(0.1f,0.0f,0.0f,1.0f);

    std::vector<glm::vec3> temp_vertices;
    res = loadOBJ("cube.obj", temp_vertices, uvs, normals);
    if(res == false)
        exit(1);
    objects[1].n_vertices = temp_vertices.size();
    objects[1].speed = make_float4(0.0f,0.0f,0.1f,1.0f);

    vertices.insert(vertices.end(), temp_vertices.begin(), temp_vertices.end());

    res = loadOBJ("cube.obj", temp_vertices, uvs, normals);
    if(res == false)
        exit(1);
    objects[2].n_vertices = temp_vertices.size();
    objects[2].speed = make_float4(0.0f,0.0f,0.2f,1.0f);

    vertices.insert(vertices.end(), temp_vertices.begin(), temp_vertices.end());

    res = loadOBJ("cone.obj", temp_vertices, uvs, normals);
    if(res == false)
        exit(1);
    objects[3].n_vertices = temp_vertices.size();
    objects[3].speed = make_float4(0.0f,0.1f,0.0f,1.0f);

    vertices.insert(vertices.end(), temp_vertices.begin(), temp_vertices.end());


    printf("Size of vertices %d\n",vertices.size());
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float4), 0, GL_DYNAMIC_DRAW);
    // glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float4), &vertices[0], GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_TRIANGLES, 0, vertices.size());
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
    if (!d_vbo_buffer)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

        // map buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

        // check result
        if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
        {
            // write file for regression test
            sdkWriteFile<float>("./data/regression.dat",
                                data, mesh_width * mesh_height * 3, 0.0, false);
        }

        // unmap GL buffer object
        if (!glUnmapBuffer(GL_ARRAY_BUFFER))
        {
            fprintf(stderr, "Unmap buffer failed.\n");
            fflush(stderr);
        }

        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsWriteDiscard));

        SDK_CHECK_ERROR_GL();
    }
}


bool loadOBJ(
    const char * path, 
    std::vector<glm::vec3> & out_vertices, 
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals
){
    printf("Loading OBJ file %s...\n", path);

    std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
    std::vector<glm::vec3> temp_vertices; 
    std::vector<glm::vec2> temp_uvs;
    std::vector<glm::vec3> temp_normals;


    FILE * file = fopen(path, "r");
    if( file == NULL ){
        printf("Impossible to open the file ! Are you in the right path ? See Tutorial 1 for details\n");
        getchar();
        return false;
    }

    while( 1 ){

        char lineHeader[128];
        // read the first word of the line
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break; // EOF = End Of File. Quit the loop.

        // else : parse lineHeader
        
        if ( strcmp( lineHeader, "v" ) == 0 ){
            glm::vec3 vertex;
            fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z );
            temp_vertices.push_back(vertex);
        }
        // else if ( strcmp( lineHeader, "vt" ) == 0 ){
        //     glm::vec2 uv;
        //     fscanf(file, "%f %f\n", &uv.x, &uv.y );
        //     uv.y = -uv.y; // Invert V coordinate since we will only use DDS texture, which are inverted. Remove if you want to use TGA or BMP loaders.
        //     temp_uvs.push_back(uv);
        // }
        // else if ( strcmp( lineHeader, "vn" ) == 0 ){
        //     glm::vec3 normal;
        //     fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z );
        //     temp_normals.push_back(normal);
        // }
        else if ( strcmp( lineHeader, "f" ) == 0 ){
            std::string vertex1, vertex2, vertex3;
            unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
            // int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2] );
            int matches = fscanf(file, "%d %d %d\n", &vertexIndex[0], &vertexIndex[1], &vertexIndex[2]);
            // if (matches != 9){
            if (matches != 3){
                printf("File can't be read by our simple parser :-( Try exporting with other options\n");
                return false;
            }
            vertexIndices.push_back(vertexIndex[0]);
            vertexIndices.push_back(vertexIndex[1]);
            vertexIndices.push_back(vertexIndex[2]);
            // uvIndices    .push_back(uvIndex[0]);
            // uvIndices    .push_back(uvIndex[1]);
            // uvIndices    .push_back(uvIndex[2]);
            // normalIndices.push_back(normalIndex[0]);
            // normalIndices.push_back(normalIndex[1]);
            // normalIndices.push_back(normalIndex[2]);
        }else{
            // Probably a comment, eat up the rest of the line
            char stupidBuffer[1000];
            fgets(stupidBuffer, 1000, file);
        }

    }

    // For each vertex of each triangle
    for( unsigned int i=0; i<vertexIndices.size(); i++ ){

        // Get the indices of its attributes
        unsigned int vertexIndex = vertexIndices[i];
        // unsigned int uvIndex = uvIndices[i];
        // unsigned int normalIndex = normalIndices[i];
        
        // Get the attributes thanks to the index
        glm::vec3 vertex = temp_vertices[ vertexIndex-1 ];
        // glm::vec2 uv = temp_uvs[ uvIndex-1 ];
        // glm::vec3 normal = temp_normals[ normalIndex-1 ];
        
        // Put the attributes in buffers
        out_vertices.push_back(vertex);
        // out_uvs     .push_back(uv);
        // out_normals .push_back(normal);
    
    }

    return true;
}

