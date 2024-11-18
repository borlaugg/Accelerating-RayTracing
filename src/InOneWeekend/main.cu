//ƒ==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "rtweekend.h"
#include <curand_kernel.h>
#include <curand.h>
#include "camera.h"
#include "hittable.h"
#include <time.h>
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

// typedef cudaError cudaError_t;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void create_world(hittable_list* world, hittable** world_dummy) {
    printf("Here\n");
    printf("%d\n", world->tail_index);
    // world_dummy[0];
    printf("%p\n", world->objects[0]);
    printf("%p\n", world_dummy[0]);
    for (int i =0; i < 488; i++){
        world->objects[i] = world_dummy[i];
    }
    
}

__global__ void render_init(camera *cam, curandState *rand_state) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int j = (blockIdx.y * blockDim.y + threadIdx.y);
    int pixel_index = i + (*cam).image_width*j;

    // printf("%d\n", pixel_index);

    if (i >= (*cam).image_width || j >= (*cam).image_height)
        return;

    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
    // printf("Done");
}

// Code that will be run on the GPU
__global__ void render(const hittable_list* world, color *pixel_color, camera *cam, curandState *rand_state) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int j = (blockIdx.y * blockDim.y + threadIdx.y);
    int pixel_index = i + (*cam).image_width*j;

    if (i >= (*cam).image_width || j >= (*cam).image_height)
        return;

    // curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);

    curandState local_rand_state = rand_state[pixel_index];
    
    // printf("Samples: %d\n", (*cam).samples_per_pixel);

    for (int sample = 0; sample < (*cam).samples_per_pixel; sample++) {
        ray r = (*cam).get_ray(i, j, &local_rand_state);
        // printf("%d %p\n", (*cam).max_depth, &local_rand_state);
        pixel_color[pixel_index] += (*cam).ray_color(r, (*cam).max_depth, *world, &local_rand_state);
        printf("%f %f %f\n", pixel_color[pixel_index].x(), pixel_color[pixel_index].y(), pixel_color[pixel_index].z());
    }
    // cudaError_t error = cudaGetLastError();
    // printf("Error: %s \n", cudaGetErrorString(error));
    
    // __syncthreads();
    
    // write_color(std::cout, (*cam).pixel_samples_scale * pixel_color[pixel_index]);
}


int main(int argc, char **argv) {
    if (argc < 3){
        printf("Format : ./main imageWidth numThreadsPerBlock_x numThreadsPerBlock_y");
    }

    int numThreadsPerBlock_x = atoi(argv[2]);
    int numThreadsPerBlock_y = atoi(argv[3]);

    hittable_list world;

    //auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    lambertian* ground_material = new lambertian(color(0.5, 0.5, 0.5)); // Replacing the above case.
    //world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));
    world.add(new sphere(point3(0,-1000,0), 1000, ground_material));    // sphere is the pointer type

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = std::rand() / (RAND_MAX + 1.0);
            point3 center(a + 0.9*std::rand() / (RAND_MAX + 1.0), 0.2, b + 0.9*std::rand() / (RAND_MAX + 1.0));

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material* sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = new lambertian(albedo);
                    world.add(new sphere(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = 0.5 * std::rand() / (RAND_MAX + 1.0);
                    sphere_material = new metal(albedo, fuzz);
                    world.add(new sphere(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = new dielectric(1.5);
                    world.add(new sphere(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = new dielectric(1.5);
    world.add(new sphere(point3(0, 1, 0), 1.0, material1));

    auto material2 = new lambertian(color(0.4, 0.2, 0.1));
    world.add(new sphere(point3(-4, 1, 0), 1.0, material2));

    auto material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
    world.add(new sphere(point3(4, 1, 0), 1.0, material3));

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = atoi(argv[1]);
    cam.samples_per_pixel = 10;
    cam.max_depth         = 20;

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    printf("camera's image width = %d\n", cam.image_width);
    printf("Number of threads per block = %d, %d\n", numThreadsPerBlock_x, numThreadsPerBlock_y);
    
    clock_t start, stop;
    start = clock();
    cam.initialize();                                               // Need to call this once before we pass things to the GPU
    
    // CUDA doesn't have std::rand() therefore we need to define a set of seds
    curandState *d_rand_state;
    printf("%d\n",cam.image_height * cam.image_height);
    cudaMalloc((void **)&d_rand_state, cam.image_height * cam.image_height * sizeof(curandState));
    
    color* pixel_color;
    cudaMalloc((void **)&pixel_color, cam.image_width * cam.image_height * sizeof(color));        // We will allocate memory in the GPU for storing the pixel colors
    
    // Allocating appropriate space in GPU
    int num_entries = world.size();
    // std::cout << num_entries << std::endl;
    hittable_list *d_world;
    cudaMalloc((void **)&d_world, sizeof(world));
    camera *d_cam;
    cudaMalloc((void **)&d_cam, sizeof(camera));

    // Copying relevant data
    cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice);
    cudaMemcpy(d_world, &world, sizeof(world), cudaMemcpyHostToDevice);

    hittable* d_list_of_spheres[488]; // We need to share this to GPU DRAM. And also each of the objects this list points to
    cudaMalloc((void **)&d_list_of_spheres, 488*sizeof(hittable*));
    checkCudaErrors(cudaGetLastError());

    for (int i = 0; i < 488; i ++){
        cudaMalloc((void **)&d_list_of_spheres[i], sizeof(hittable));
        cudaMemcpy(d_list_of_spheres[i], world.objects[i], sizeof(hittable), cudaMemcpyHostToDevice);
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // hittable_list *h_world_why_not = (hittable_list *)malloc(sizeof(world));
    // memcpy(h_world_why_not, &world, sizeof(world));

    // printf("%d\n", h_world_why_not->tail_index);

    dim3 blocks(cam.image_height/numThreadsPerBlock_x + 1,cam.image_width/numThreadsPerBlock_y + 1);
    dim3 threads(numThreadsPerBlock_x, numThreadsPerBlock_y);
    
    create_world<<<1, 1>>>(d_world, d_list_of_spheres);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // std::cout << world.tail_index << std::endl;
    render_init<<<blocks, threads>>>(d_cam, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(d_world, pixel_color, d_cam, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();

    // printf("Rnedingnineign\n");

    color* h_pixel_color = (color*)malloc(cam.image_width * cam.image_height * sizeof(color));
    cudaMemcpy(h_pixel_color, pixel_color, cam.image_width * cam.image_height * sizeof(color), cudaMemcpyDeviceToHost);
    for (int j = 0 ; j < cam.image_height; j++){
        for (int i = 0 ; i < cam.image_width; i++){
            write_color(std::cout, cam.pixel_samples_scale * h_pixel_color[i + cam.image_width*j]);
        }
    }

    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << timer_seconds << "\n";

}
