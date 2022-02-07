#!/bin/bash
cd ../..
source carla-env/bin/activate
python run_simulation.py -p 2004 --start=11 --end=11 --record_mode=2 --script_name=$(echo ${0##*/} | cut -d'.' -f1)