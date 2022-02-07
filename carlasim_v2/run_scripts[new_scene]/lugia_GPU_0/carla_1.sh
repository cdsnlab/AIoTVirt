#!/bin/bash
SDL_HINT_CUDA_DEVICE=0 /home/tung/carla/CARLA_Shipping_0.9.11-dirty/LinuxNoEditor/CarlaUE4.sh -carla-settings=Example.CarlaSettings.ini -windowed -resx=480 -resy=360 -opengl -world-port=2000
#SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 CUDA_VISIBLE_DEVICES=0 ./CarlaUE4.sh -carla-settings=Example.CarlaSettings.ini -windowed -resx=480 -resy=360 -opengl -world-port=2001
#../../CarlaUE4.sh -carla-settings=Example-CarlaSetting.ini -windowed -resx=480 -resy=360 -opengl
