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

#include "camera.h"
#include "hittable.h"
#include <time.h>
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

// Code that will be run on the GPU
__global__ void render(const hittable* world, color *pixel_color, camera *cam, curandState *rand_state) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int j = (blockIdx.y * blockDim.y + threadIdx.y);
    int pixel_index = i + (*cam).image_width*j;

    if (i >= (*cam).image_width || j >= (*cam).image_height)
        return;

    curand_init(1947 + pixel_index, 0, 0, &rand_state[pixel_index]);

    curandState local_rand_state = rand_state[pixel_index];
    
    for (int sample = 0; sample < (*cam).samples_per_pixel; sample++) {
        ray r = (*cam).get_ray(i, j, &local_rand_state);
        pixel_color[pixel_index] += (*cam).ray_color(r, (*cam).max_depth, *world, &local_rand_state);
    }
    
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

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = std::rand() / (RAND_MAX + 1.0);
            point3 center(a + 0.9*std::rand() / (RAND_MAX + 1.0), 0.2, b + 0.9*std::rand() / (RAND_MAX + 1.0));

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = 0.5 * std::rand() / (RAND_MAX + 1.0);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

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

    return 0;
    
    // CUDA doesn't have std::rand() therefore we need to define a set of seds
    curandState *d_rand_state;
    cudaMalloc((void **)&d_rand_state, cam.image_height * cam.image_height * sizeof(curandState));
    
    clock_t start, stop;
    start = clock();
    cam.initialize();                                               // Need to call this once before we pass things to the GPU
    color* pixel_color;
    cudaMalloc((void **)&pixel_color, cam.image_width * cam.image_height * sizeof(color));        // We will allocate memory in the GPU for storing the pixel colors
    
    // Allocating appropriate space in GPU
    int num_entries = world.size();
    hittable_list *d_world;
    cudaMalloc((void **)&d_world, num_entries * sizeof(hittable));
    camera *d_cam;
    cudaMalloc((void **)&d_cam, sizeof(camera));

    // Copying relevant data
    cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice);
    cudaMemcpy(d_world, &world, num_entries * sizeof(hittable), cudaMemcpyHostToDevice);
    
    dim3 blocks(cam.image_height/numThreadsPerBlock_x + 1,cam.image_width/numThreadsPerBlock_y + 1);
    dim3 threads(numThreadsPerBlock_x, numThreadsPerBlock_y);

    render<<<blocks, threads>>>(d_world, pixel_color, d_cam, d_rand_state);
    cudaDeviceSynchronize();
    stop = clock();

    color* h_pixel_color;
    cudaMemcpy(h_pixel_color, pixel_color, cam.image_width * cam.image_height * sizeof(color), cudaMemcpyDeviceToHost);
    for (int j = 0 ; j < cam.image_height; j++){
        for (int i = 0 ; i < cam.image_width; i++){
            write_color(std::cout, cam.pixel_samples_scale * h_pixel_color[i + cam.image_width*j]);
        }
    }

    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << timer_seconds << "\n";

}
