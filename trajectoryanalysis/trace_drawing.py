import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--input_folder', default='/home/spencer1/AIoTVirt/trajectoryanalysis/npy/connected_nodup', 
                    type=str, help='Folder where the NPY files are stored')
parser.add_argument('-i', '--image_folder', default='/home/boyan/AIoTVirt/data/backgrounds/', type=str, help='Folder where the image backgrounds are stored')
args = parser.parse_args()

    
def load_npy(filename):
    return np.load(filename, allow_pickle=True)

colors = {
    0: 'k',
    1: 'r',
    2: 'g',
    3: 'b',
    4: 'c',
    5: 'm',
    6: 'y',
    7: '#B3446C',
    8: '#E25822',
    9: '#2B3D26'
}


def draw_on_image(datapath, imgpath, camera=-1):
    data = load_npy(datapath)
    plt.clf()
    print(data.shape)
    for path, label, duration, transition in data:
        if camera == -1 or label == camera:
            x, y = (*zip(*path),)
            xx = range(len(x))
            lwidths = (np.array(xx) / len(xx)) * 3
            plt.scatter(x,y, s=lwidths, color=colors[label], label=f"Camera {label}")
    plt.xlim(0, 2560)
    plt.ylim(0, 1400)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.gca().invert_yaxis()
    img = plt.imread(imgpath)
    plt.imshow(img,zorder=0, alpha=0.5)
    plt.savefig(imgpath.replace('cam', 'trace_vis_cam_'), dpi=500)
    return plt

for i in range(10):
    draw_on_image("{}/{}.npy".format(args.input_folder, i), '{}/cam{}.png'.format(args.image_folder, i))



