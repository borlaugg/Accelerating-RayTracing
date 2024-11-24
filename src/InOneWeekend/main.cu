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

long material_size(material* mat) {
    if (metal* d = dynamic_cast<metal*>(mat)) {
        return sizeof(metal);
    } 
    else if (lambertian* d = dynamic_cast<lambertian*>(mat)) {
        return sizeof(lambertian);
    } 
    else if (dielectric* d = dynamic_cast<dielectric*>(mat)) {
        return sizeof(dielectric);
    }
    return sizeof(*mat);
}

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void create_world(hittable_list* world, sphere* sphere_list) {
    for (int i =0; i < world->size(); i++){
        world->objects[i] = &sphere_list[i];
    }
}

__global__ void render_init(camera *cam, curandState *rand_state) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int j = (blockIdx.y * blockDim.y + threadIdx.y);
    int pixel_index = i + (*cam).image_width*j;

    if (i >= (*cam).image_width || j >= (*cam).image_height)
        return;

    curand_init(1947 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__device__ bool hit_world(const hittable_list* world, ray& r, interval ray_t, hit_record& rec) {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;
    for (const auto& object: world->objects) {
        if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ vec3 ray_color(const ray& r, const hittable_list* world, curandState* state) {

    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);

    for(int i = 0; i < 50; i++) {
        printf("%d\n", i);
        hit_record rec;
        if (hit_world(world, cur_ray, interval(0.001, infinity), rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat->scatter(cur_ray, rec, attenuation, scattered, state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

// Code that will be run on the GPU
__global__ void render(const hittable_list *world, camera *cam, curandState *randState, color *pixel_color) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int j = (blockIdx.y * blockDim.y + threadIdx.y);
    int pixel_index = i + (*cam).image_width*j;

    if (i >= (*cam).image_width || j >= (*cam).image_height)
        return;

    curandState local_randState = randState[pixel_index];
    
    for (int sample = 0; sample < (*cam).samples_per_pixel; sample++) {
        ray r = (*cam).get_ray(i, j, &local_randState);
        // printf("%d %p\n", (*cam).max_depth, &local_randState);
        pixel_color[pixel_index] += ray_color(r, world, &local_randState);
        // pixel_color[pixel_index] += (*cam).ray_color(r, (*cam).max_depth, world, &local_randState);
    }
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
    cam.max_depth         = 2;

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;
    // printf("world.size() = %d\n", world.size());
    // printf("camera's image width = %d\n", cam.image_width);
    // printf("Number of threads per block = %d, %d\n", numThreadsPerBlock_x, numThreadsPerBlock_y);
    
    clock_t start, stop;
    start = clock();
    cam.initialize();                                               // Need to call this once before we pass things to the GPU
    
    // CUDA doesn't have std::rand() therefore we need to define a set of seds
    void* d_randState;         // Abse we will start using pointers as just that -- pointers. We don't think of pointers of being of a specifc class
    cudaMalloc((void **)&d_randState, cam.image_width * cam.image_height * sizeof(curandState));
    
    void* d_pixel_color;                                  // Making it a pointer for the sake of my sanity.
    cudaMalloc((void **)&d_pixel_color, cam.image_width * cam.image_height * sizeof(color));        // We will allocate memory in the GPU for storing the pixel colors
    
    // Allocating appropriate space in GPU
    int num_entries = world.size();
    
    void* d_cam;                // Making it a pointer for the sake of my sanity.
    cudaMalloc((void **)&d_cam, sizeof(camera));
    // Copying relevant data
    cudaMemcpy(d_cam, (void *)&cam, sizeof(camera), cudaMemcpyHostToDevice);

    void* d_list_of_spheres; // It is just a pointer for now. It points to the memory location which contiguously contains all of the objects
    cudaMalloc((void **)&d_list_of_spheres, world.size() * sizeof(sphere));
    checkCudaErrors(cudaGetLastError());

    // Sharing every sphere with the GPU. Note that we also need to send the material to GPU
    void* material_ptr;
    // std::cout << "Size of lambertian = " << sizeof(lambertian) << std::endl;
    // std::cout << "Size of dielectric = " << sizeof(dielectric) << std::endl;
    // std::cout << "Size of metal = " << sizeof(metal) << std::endl;
    // std::cout << "Size of material = " << sizeof(material) << std::endl;

    for (int i = 0; i < world.size(); i ++){        
        // Let us first allocate space for the material that makes the sphere.
        cudaMalloc((void **)&material_ptr, material_size((*world.objects[i]).mat));
        // Let us now copy the material
        cudaMemcpy(material_ptr, (*world.objects[i]).mat, material_size((*world.objects[i]).mat), cudaMemcpyHostToDevice);
        // Let us update the sphere's pointer to material before we send it over
        world.objects[i]->mat = (material *)material_ptr;
        // Sending the material now fr
        cudaMemcpy((void *)&(((sphere *)d_list_of_spheres)[i]), (void *)world.objects[i], sizeof(sphere), cudaMemcpyHostToDevice);
        // Let us now update the pointer to the sphere in the world
        world.objects[i] = &(((sphere *)d_list_of_spheres)[i]);
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Now we will share the object of class hittable_list with the GPU
    void* d_world;
    cudaMalloc((void **)&d_world, sizeof(hittable_list));
    cudaMemcpy(d_world, &world, sizeof(hittable_list), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // Created a world on the GPU and we even have the location of that as a pointer

    // This will update the world object in the GPU to accurately point to the actual spheres
    // create_world<<<1, 1>>>((hittable_list *)d_world, (sphere *)d_list_of_spheres);         // The sole job of this is to create the world using the pointer to the list and the update the d_world pointer accordingly
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(cam.image_height/numThreadsPerBlock_x + 1,cam.image_width/numThreadsPerBlock_y + 1);
    dim3 threads(numThreadsPerBlock_x, numThreadsPerBlock_y);
    
    render_init<<<blocks, threads>>>((camera *)d_cam, (curandState *)d_randState);     // We are passing pointers to the camera object, rand_state and world -- all of these are already in the GPU memory
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>((hittable_list *)d_world, 
                                (camera *)d_cam, 
                                (curandState *)d_randState, 
                                (color *)d_pixel_color);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();

    color* h_pixel_color = (color*)malloc(cam.image_width * cam.image_height * sizeof(color));
    cudaMemcpy(h_pixel_color, d_pixel_color, cam.image_width * cam.image_height * sizeof(color), cudaMemcpyDeviceToHost);
    for (int j = 0 ; j < cam.image_height; j++){
        for (int i = 0 ; i < cam.image_width; i++){
            write_color(std::cout, cam.pixel_samples_scale * h_pixel_color[i + cam.image_width*j]);
        }
    }

    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    // std::cerr << timer_seconds << "\n";

}
