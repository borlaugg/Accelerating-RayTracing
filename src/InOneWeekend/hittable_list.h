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

#include <vector>


class hittable_list : public hittable {
  public:
    hittable* objects[1 + 22 * 22 + 3];               // We will use this array of 100 pointers
    int tail_index;

    __device__ __host__ hittable_list() { 
        tail_index = 0;
    }
    __device__ __host__ hittable_list(hittable* object) { 
        tail_index = 0;
        add(object); 
    }

    __device__ __host__ void clear() { 
        for (int i =0; i < tail_index; i++){
            objects[i] = nullptr;
        }
    }

    __device__ __host__ void add(hittable* object) {
        objects[tail_index] = object;
        // printf("Added this mf : %p", object);
        tail_index ++;
    }

    int size() {
        return tail_index;
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        printf("Here");
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (const auto& object: objects) {
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                // printf("Here2");
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        // printf("%d\n", hit_anything);
        return hit_anything;
    }
};


#endif
