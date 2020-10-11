'''
[Prog]
This program reads gt files from AIC, 
draws relationship btw cameras in terms of directed graphs. 
'''
CAMERA_SECTION ={ #* S05 is a mix of cameras from Section 03 to 04
    #* train folder
    "S01": {"c001", "c002", "c003", "c004", "c005"},
    "S03": {"c010", "c011", "c012", "c013", "c014", "c015"},
    "S04": {"c016", "c017", "c018", "c019", "c020", "c021", "c022", "c023", "c024", "c025", "c026", "c027", "c028", "c029", "c030", "c031", "c032", "c033", "c034", "c035", "c036", "c037", "c038", "c039", "c040"},
    #* validation folder
    "S02": {"c006", "c007", "c008", "c009"},
    "S05": {"c010", "c016", "c017", "c018", "c019", "c020", "c021", "c022", "c023", "c024", "c025", "c026", "c027", "c028", "c029", "c033", "c034", "c035", "c036"}
    #* test folder
    #"S06": {"c041", "c042", "c043", "c044", "c045", "c046"} #! does not contain gt
}

import networkx as nx
from tqdm import tqdm
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt

client = MongoClient('localhost', 27017)
db = client['aic_mtmc']
iddb = db['uid_traces']
stalldb = db['st_all']

# 1) read all pairs
# 2) draw all pairs


def get_pairs():
    stitems={}
    srclookup={}
    dupchecker=[]
    doc=list(stalldb.find(srclookup, {"_id":0, "pairs": 1, "src":1, "dst": 1, "temporal":1}))
    i=0
    for d in doc:
        if (d['src'], d['dst']) not in dupchecker:
            dupchecker.append((d['src'], d['dst']))
            stitems[i] = (d['src'], d['dst'])
            i+=1
        else:
            print("dup found")

    #! check for duplications...
    return stitems

allpairs = get_pairs()
# print(allpairs)
plt.figure(figsize=(18,18))
G = nx.DiGraph()
for ap in allpairs:
    print(allpairs[ap])
    G.add_edges_from([allpairs[ap]])
pos = nx.spring_layout(G, k=0.7)
# pos = nx.spring_layout(G, k=0.7, iterations=100)
nx.draw(G, pos, font_size=16, node_size=5, with_labels=True)
# nx.draw_circular(G, pos, font_size=10, node_size=1, with_labels=True)
plt.savefig("test.jpg")
