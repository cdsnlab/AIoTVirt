from torchreid.utils import FeatureExtractor
import time
from torchreid import metrics
import glob
import threading
import sys
from multiprocessing import Process
extractor = FeatureExtractor(
    # model_name='mobilenetv2_x1_4',
    model_name='osnet_x1_0',
    model_path='trajectoryanalysis/reid/models/osnet_x1_0_imagenet.pth',
    # model_path='trajectoryanalysis/reid/models/mobilenetv2_1.4.pth',
    device='cpu'
)

# image_list = [ 'carlasim/data/CARLA/cam_0_frame_{}.jpg'.format(i) for i in range(3004, 3028)]
image_list = glob.glob("/home/boyan/AIoTVirt/carlasim/data/CARLA/*.jpg")

crowd = ['carlasim/data/static_people/cam_0_person_{}.jpg'.format(person) for person in range(12)]

crowd_2 = ['carlasim/data/static_people/cam_2_person_{}.jpg'.format(person) for person in range(26, 39)]
crowd_2 = crowd + crowd_2

total_time = 0

carla = extractor(['carlasim/data/CARLA/CARLA.jpg'])

def descriptors(images):
    global carla
    # start = time.time()
    features = extractor(images)
    metrics.compute_distance_matrix(carla, features, metric="cosine")
    # total_time += time.time() - start
    
print(threading.activeCount())
    
repeats = 20
cameras = int(sys.argv[1])
results = 0
# tt = 0
sss =time.time()
results = []
# for cameras in range(9, 10):
tt = 0
for j in range(repeats):
    # descriptors(image_list[:15])
    threads = []
    start = time.time()
    descriptors(image_list[:15])
    # for i in range(cameras):
    #     x = threading.Thread(target=descriptors, args=(image_list[:100],))
    #     threads.append(x)
    #     x.start()
        
    # for index, thread in enumerate(threads):
    #     thread.join()
        
    end = time.time()
    tt += end - start
results.append(tt / repeats)
time.sleep(2)
# print((time.time() - sss) / repeats) 
# print(tt / repeats)
# print(total_time / repeats)
print(results)
    
  
# crowd_2 = glob.glob("/home/boyan/AIoTVirt/carlasim/data/static_people/*.jpg")
# start = time.time()
# features = extractor(image_list)
# start = time.time()
# features = extractor(image_list[:15])
# print(time.time() - start)
# # print(features.shape) # output (5, 512)

# start = time.time()
# carla = extractor(['carlasim/data/CARLA/CARLA.jpg'])
# print(time.time() - start)
# carla_2 = extractor(['carlasim/data/CARLA/CARLA_2.jpg'])
# carla_3 = extractor(['carlasim/data/CARLA/CARLA_3.jpg'])

# c = extractor(crowd)
# c2 = extractor(crowd_2)
# results = []
# # carla = features[0]
# # for person in features[1:]:
# #     results.append(metrics.compute_distance_matrix(carla, person))
# #     print(carla.shape)
# #     print(person.shape)
#     # results.append(metrics.distance.cosine_distance(carla, person))
    
# start = time.time()    
# print("CARLAS ", metrics.compute_distance_matrix(carla, features, metric="cosine").mean())
# # print(time.time() - start)
# start = time.time()    
# # print("CROWD CAM_0: ", metrics.compute_distance_matrix(carla, c2, metric="cosine"))
# print("CROWD CAM_1: ", metrics.compute_distance_matrix(carla, c2, metric="cosine").mean())
# # start = time.time()    
# print("COSINE CARLA Frame n - n+1: ", metrics.compute_distance_matrix(carla_3, carla_2, metric="cosine"))
# # print(time.time() - start)
# # print("EUCLID CARLA Frame n - n+1: ", metrics.compute_distance_matrix(carla_3, carla_2, metric="euclidean"))
# # print(results)