Things we can try to do: 
1. Make `render()` as a `__global__` function
2. Make `get_ray()` a device function.
3. `initialise()` need not be called by every thread. Therefore, we will run it separately. Since it is a private method, let's first make it public.