Things we can try to do: 
1. Make `render()` as a `__global__` function
2. Make `get_ray()` a device function.
3. `initialise()` need not be called by every thread. Therefore, we will run it separately. Since it is a private method, let's first make it public.
4. Effectively, the current bottleneck is due to `make_shared` and `shared_ptr` that are taken from the std library. We can replace them easily because the most they do is make the operation efficienct wrt space occupied.
However, it isn't as easy as just replacing it with some bullshit. We will need to ensure that we use vectors for whatever reason the code uses them. Turns out there is the `thrust` library that allows us to easily make such vectors but like wow. I am not cut out for this mehnat rn.
