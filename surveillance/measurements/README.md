# 2018-03-09

The measurements ending with "raw" are as they are saved from the
processing threads, which includes a call to ps from within the process and
capturing its output with python. The measurements are made approx. each second.

The csv files are basically the same information but better organized. They
only contain information related to one process (either face or eye
detection). They do not include measurements from the camera thread nor the
main thread dispatching the frames.

The multiapp_meass.csv file contains the processed data. AvgPSR refers to
the amount of cpu cores (processors/psr) that were being used in avg by all
the threads (Since the threads are not pinned they can run in an arbitrary
core and jump quite a lot. They can all run in the same core, all in
different cores or somewhere in between). The AvgCPU column shows how much
_aggregated_ CPU was used by a process (sum of all cores). AvgCPU/PSR
indicates how much CPU _per core_ was used; This column is derived by
dividing AvgCPU / AvgPSR

My processing steps are in the jupyter notebook
