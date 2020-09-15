import pandas as pd
import matplotlib.pyplot as plt
#from typing import NamedTuple
import numpy as np
import os
import time, math
import glob
import statistics
'''
this program evaluates the lengths of the csvs
'''

def get_transition_dist(path, naive=False):
    filenames = os.listdir(path)
    lengths = []
    for file in filenames:
        if '.csv' in file:
            #print(file)
            data = pd.read_csv(path + "/" + file)
            #print(len(data))
            lengths.append(len(data))
    print(statistics.mean(lengths))
    print(statistics.stdev(lengths))


get_transition_dist('/home/spencer1/samplevideo/new_sim_csv/', False)
