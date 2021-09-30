#!/bin/bash
cd '/home/tung/carla/CARLA_Shipping_0.9.11-dirty/LinuxNoEditor/PythonAPI/carlasim'
source carla-env/bin/activate
python run_simulation.py -p 2012 --start=890 --end=900 --record_mode=2 --script_name=$(echo ${0##*/} | cut -d'.' -f1)