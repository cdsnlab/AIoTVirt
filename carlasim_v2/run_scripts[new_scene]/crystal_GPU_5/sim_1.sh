#!/bin/bash
cd '/home/tung/carla/CARLA_Shipping_0.9.11-dirty/LinuxNoEditor/PythonAPI/carlasim'
source carla-env/bin/activate
python run_simulation.py -p 2008 --start=53 --end=57 --num_walkers=53 --record_mode=2 --script_name=$(echo ${0##*/} | cut -d'.' -f1)