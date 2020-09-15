from scipy.io import loadmat
annots = loadmat('/home/spencer1/samplevideo/RPIfield_Info/Cam_10.mat')
print(annots.keys())
print(annots.values())
