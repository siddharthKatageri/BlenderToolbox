This is a set of scripts for rendering 3D shapes in Blender. To use them, make sure you have installed Blender 2.80 and you can run the demo by typing 
```
blender --background --python demo.py
```

Before rendering a scene, a very important step is to set up the default rendering devices in the user preferences (e.g., which GPU to  use). You only need to save the user preferences once, then the script should detect the GPU on the computer and use them for rendering. To set up the rendering devices, open the blender, go to `Edit` > `Preferences` > `System`, then in the `Cycles Render Devices` select your preferred devices for rendering (e.g., select `CUDA` and check every GPUs and CPUs on your computer). After selecting set up the devices, click the `Save  Preference` on bottom left. If you forget to set up the preference, blender may not be using GPU for rendering.