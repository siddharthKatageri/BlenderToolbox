import sys
import os
sys.path.append(os.getcwd()) # change this to your path to “path/to/BlenderToolbox/"
import BlenderToolBox as bt
import os, bpy, bmesh
import numpy as np
import re

cwd = os.getcwd()

def hex2rgb(h):
    return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def set_gpu(cuda_no, scene):
    scene.cycles.device = "GPU"
    prefs = bpy.context.preferences
    prefs.addons["cycles"].preferences.get_devices()
    cprefs = prefs.addons["cycles"].preferences
    # Attempt to set GPU device types if available
    for compute_device_type in ("CUDA", "OPENCL", "NONE"):
        try:
            cprefs.compute_device_type = compute_device_type
            print("Device found", compute_device_type)
            break
        except TypeError:
            pass
    scene.cycles.use_auto_tile = True
    #  for scene in bpy.data.scenes:
    #  scene.render.tile_x = 256
    #  scene.render.tile_y = 256
    # Enable all CPU and GPU devices
    for device in cprefs.devices:
        print(device)
        if not re.match("intel", device.name, re.I):
            print("Activating", device)
            device.use = False
        else:
            device.use = False
    for n in cuda_no:
        cprefs.devices[n].use = True


colors = [
    '#1f77b4',  # muted blue 0
    '#ff7f0e',  # safety orange 1
    '#2ca02c',  # cooked asparagus green 2
    '#d62728',  # brick red 3
    '#9467bd',  # muted purple 4
    '#8c564b',  # chestnut brown 5
    '#e377c2',  # raspberry yogurt pink 6
    '#7f7f7f',  # middle gray 7
    '#bcbd22',  # curry yellow-green 8
    '#17becf',  # blue-teal 9
    '#8ea8ff'   # blue 10
]


colors = np.array([[float(c)/255.0 for c in hex2rgb(color)] for color in colors])
colors = np.hstack((colors, np.ones((colors.shape[0], 1))))


dataset="modelnet10"
type="intra"
folder="euc_240_250"
name = "pc_recon240_modelnet_10_supcon_recon_cd_alp0.2_bs16_euc_dist_ground1"

outputPath = os.path.join(cwd, '../diagrams/interpolate/',dataset,type,folder,name+'.png') # make it abs path for windows

## initialize blender
imgRes_x = 1080 # recommend > 1080 (UI: Scene > Output > Resolution X)
imgRes_y = 1080 # recommend > 1080 
numSamples = 200 # recommend > 200 for paper images
exposure = 1.5 
bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
bpy.context.scene.cycles.device = 'GPU'


## read mesh (choose either readPLY or readOBJ)
meshPath = os.path.join('../diagrams/interpolate',dataset,type,folder, name+'.ply')

# location = (1.12, -0.14, 0) # (UI: click mesh > Transform > Location)
location = (0.92, -0.21, 1.26) # (UI: click mesh > Transform > Location)
rotation = (90, 0, -130) # (UI: click mesh > Transform > Rotation)
scale = (1.5,1.5,1.5) # (UI: click mesh > Transform > Scale)
mesh = bt.readMesh(meshPath, location, rotation, scale)

## Draw points
# RGBA = (144.0/255, 210.0/255, 236.0/255, 1)
print(colors.shape)
# RGBA = colors[10]
RGBA = np.array([0.27, 0.392, 1, 1]) # fixing bluish-purple rgba color taken from blender
ptColor = bt.colorObj(RGBA, 0.5, 1.3, 1.0, 0.0, 0.0)
ptSize = 0.02
bt.setMat_pointCloud(mesh, ptColor, ptSize)

## set invisible plane (shadow catcher)
bt.invisibleGround(shadowBrightness=0.8)

## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
# camLocation = (3, 0, 2)
# camLocation = (6.29, -0.44, 3.86)
# lookAtLocation = (0,0,0.5)
# focalLength = 45 # (UI: click camera > Object Data > Focal Length)
# cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
## Option 2: if you really want to set camera based on the values in GUI, then
camLocation = (6.29, -0.44, 3.86)
rotation_euler = (60,0.3,86.5)
focalLength = 45
cam = bt.setCamera_from_UI(camLocation, rotation_euler, focalLength = focalLength)


## set light
## Option1: Three Point Light System 
# bt.setLight_threePoints(radius=4, height=10, intensity=1700, softness=6, keyLoc='left')
## Option2: simple sun light
lightAngle = (6, -30, -155) 
# strength = 2
strength = 5
# shadowSoftness = 0.3
shadowSoftness = 0.9
sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

## set ambient light
bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 


## set gray shadow to completely white with a threshold (optional but recommended)
# bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')
bt.shadowThreshold(alphaThreshold = 0.2, interpolationMode = 'CARDINAL')


## save blender file so that you can adjust parameters in the UI
bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/../diagrams/interpolate/inter_try.blend')

## save rendering
bt.renderImage(outputPath, cam)
