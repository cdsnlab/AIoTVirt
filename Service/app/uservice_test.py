import microservices as ms
import numpy as np
# from jtop import jtop
# from gevent import monkey
# monkey.patch_all(socket=False)

import time

# * Microservice from string, to be used in web server
# m = ms.utils.ms_from_str("Crop",[])
# m.kill_thread()
# vid_url = "rtsp://cdsnlab:CdsnLab7759@143.248.55.237:28556/ds-test"
vid_url = "sample_720p.mp4"
# d = Decoder(("Decoder", False, False, True), "rtsp://cdsnlab:CdsnLab7759@143.248.55.237:28556/ds-test")
decoder_parent_kwargs = {"is_head": True}
# for dev in ['cuda']:
#     for bat in [1,2,4,8,16,32,64]:

# * Build pipeline
d = ms.Decoder(
    vid_url, False, [272, 480], [2, 1, 0], opencv=True, parent_kwargs=decoder_parent_kwargs)

i = ms.Inference('resnet10nano_trt', '143.248.55.237:28000', separate_requests=True)
postp_kwargs = {'img_size': (272, 480), 'num_class': 4, 'threshold': 0.3}
p = ms.Postprocess('DetectNet', postp_kwargs)
# c = ms.Crop()
c = ms.Crop(parent_kwargs={"is_sink": True})
engine = "triton"
# triton_args = {'url': '143.248.55.237:28000', 'concurrency': 1}
triton_args = {'url': '143.248.53.69:8000', 'concurrency': 1}
r = ms.Reid('osnet_pt', 'cpu', engine=engine,
            gallery=np.zeros((1, 3, 64, 64), dtype=np.float32),
            batch=16, triton_args=triton_args, parent_kwargs={"is_sink": True})
# r = ms.Reid('osnet_x0_25', 'cuda', engine='torchreid', batch=1)

# * Link pipeline
d.link(i)
i.link(p)
p.link(c)
# c.link(r)

components = [d, i, p, c, r]

run_jtop = True

def run():
    d.start()

def stop():
    d.kill_thread()
    i.kill_thread()
    p.kill_thread()
    c.kill_thread()

# # * Run pipeline
run()

time.sleep(20)      

stop()

# * ### Print Results ###

# print("### PIPELINE TIME ### ", c.pipeline_time / c.pipeline_iters)
# # * Async numbers
# if engine == "async":
#     print(r.infertime, r.inferiters,
#           r.infertime / r.inferiters)
# # * Non-async numbers
# else:
#     # print(r.totaltime, r.inferiters,
#     #       r.totaltime / r.inferiters)
#     print(r.infertime, r.inferiters,
#           r.infertime / r.inferiters)

print("Inf per frame", i.infertime / i.inferiters)
print(c.pipeline_times)
# print(r.pipeline_time / r.pipeline_iters)
# r.kill_thread()
# print(r.pipeline_times)
