What all things do we need to push to the other side?

1. The object of class hittable_list
2. The list of pointers it contains. That list of pointers is made for host therefore we will need to update it to fit the 
3. The objects from class hittable that are being pointed to. 

Thus we should first push the entire set of objects, make a list on the side of the device which will point towards each of these objects and then either make the class on the GPU or update the class on the GPU.