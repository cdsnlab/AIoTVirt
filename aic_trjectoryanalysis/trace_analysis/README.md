## Description

### gt_to_mdb.py: 
- saves AIC2020 MTMC gt files into [aic_mtmc][draw_traces] MongoDB for plotting (Note that AIC2020 does not provide gt for test directories.)
### flow_to_mdb.py:
- traces of each UID is save to [aic_mtmc][uid_traces]
### save_bg_scene.py: 
- takes a snapshot of the view of each camera
### visualize_traces.py: 
- pulls all gt traces from [aic_mtmc][uid_traces] MongoDB and plots them on top of each camera views
### spatio_temporal.py : 
- finds relationship btw cameras in terms of time and pairs and saved in [aic_mtmc][spatio_temporal] MongoDB

Do run the following command line to put gt files in to mongodb for later use. 
```
awk '{print "c005, S01, " $0}' ~/samplevideo/AIC20_track3_MTMC/train/S01/c005/gt/gt.txt >> ~/samplevideo/AIC20_track3_MTMC/train/S01/c005/gt/gt.out
```

```
 mongoimport --db=aic_mtmc --collection=gt --type=csv --columnsHaveTypes --fields="camera.string(), sector.string(), framenumber.int32(), id.int32(), xmin.int32(), ymin.int32(), width.int32(), height.int32(), holder1.int32(), holder2.int32(), holder3.int32(), holder4.int32()" --file= ~/samplevideo/AIC20_track3_MTMC/train/S01/c001/gt/gt.out
```
<span style="color:red">Cameras 41 ~ 46 does not have ground truth data and vehicles from these cameras don't appear in train set. </span>
