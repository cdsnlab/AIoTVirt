"""
[Progress]
this program finds the matchings between cameras by reading the db.
1) which cameras has relationship btw each other. (spatio)
2) how much time it takes for a car to travel from one to another. (temporal)
"""

import os, sys, time
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from pymongo import MongoClient


#* connect to mongodb
client = MongoClient('localhost', 27017)
db = client['aic_mtmc']
mdb = db['draw_traces']
