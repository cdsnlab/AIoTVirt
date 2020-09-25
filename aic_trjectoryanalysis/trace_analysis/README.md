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


<span style="color:red">Cameras 41 ~ 46 does not have ground truth data. </span>
