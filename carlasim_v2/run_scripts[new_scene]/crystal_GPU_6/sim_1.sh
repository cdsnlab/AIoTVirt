#!/bin/bash
cd '/home/tung/carla/CARLA_Shipping_0.9.11-dirty/LinuxNoEditor/PythonAPI/carlasim'
source carla-env/bin/activate
python run_simulation.py -p 2000 --start=58 --end=62 --num_walkers=67 --record_mode=2 --script_name=$(echo ${0##*/} | cut -d'.' -f1)