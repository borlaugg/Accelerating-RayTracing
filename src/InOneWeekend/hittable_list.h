    #ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "hittable.h"
#include "sphere.h"

#include <vector>


class hittable_list {
  public:
    sphere* objects[1 + 22 * 22 + 2];               // We will use this array of 100 pointers
    int tail_index;

    __device__ __host__ hittable_list() { 
        tail_index = 0;
    }
    __device__ __host__ hittable_list(sphere* object) { 
        tail_index = 0;
        add(object); 
    }

    __device__ __host__ void clear() { 
        for (int i =0; i < tail_index; i++){
            objects[i] = nullptr;
        }
    }

    __device__ __host__ void add(sphere* object) {
        objects[tail_index] = object;
        // printf("Added this mf : %p", object);
        tail_index ++;
    }

    __device__ __host__ void balls(const ray& r){
        printf("Balls balls with this ray\n");
        printf("Ray's direction in x is %f\n", r.dir.e[0]);
    }

    __device__ __host__ void balls(hit_record& rec){
        printf("Balls balls with this record\n");
        printf("Record' value of t is %f\n", rec.t);
    }

    __device__ __host__ void balls(){
        printf("Balls balls with this world\n");
        printf("Pointer in hand is %p\n", objects[0]);
    }

    __device__ __host__ void balls(interval ray_t){
        printf("Balls balls with this interval\n");
        printf("Interval has max and min as %f and %f\n", ray_t.max, ray_t.min);
    }

    __device__ __host__ void balls(const ray& r, interval ray_t, hit_record& rec){
        printf("Balls balls with this tuple\n");
        printf("Interval has max and min as %f and %f\n", ray_t.max, ray_t.min);
        printf("Record' value of t is %f\n", rec.t);
        printf("Ray's direction in x is %f\n", r.dir.e[0]);
    }

    __device__ __host__ bool hit_world(const ray& r, interval ray_t, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;
        
        for (const auto& object: objects) {
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                printf("Here for partying\n");
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        printf("Did we hit anything? %d\n", hit_anything);
        return hit_anything;
    }

    __device__ __host__ int size() {
        return tail_index;
    }
};


#endif
