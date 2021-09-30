import microservices as ms
import cv2
import time
import numpy as np

decoder_parent_kwargs = {"is_head": True}

d = ms.Decoder("sample_720p.h264", False, [272, 480], [2, 1, 0], decoder_parent_kwargs)
# d = ms.Decoder("rtsp://cdsnlab:CdsnLab7759@143.248.55.237:28556/ds-test",(272,480),[2, 1, 0], decoder_parent_kwargs)

# r = ms.Resize([272, 480], "np.float32")
# c = ms.Crop("topleft-botright", [[50, 50, 20, 20]])
# t = ms.Transpose([2, 1, 0])
infer_kwargs = {'model': 'resnet10nano_trt', 'infer_url': 'localhost:8000'}
i = ms.Inference(**infer_kwargs)
postprocess_kwargs = {'img_size': (272, 480), 'num_class': 4}
p = ms.Postprocess('DetectNet', postprocess_kwargs)

# d.link(r)
# r.link(t)
# t.link(i)
d.link(i)
i.link(p)
a = time.time()
d.start()
for _ in range(60):
    time.sleep(1)
    # print(d.totaltime/ d.iters,d.totaltime, d.iters)
    # print('Decoder ',d.totaltime/ d.iters,'\n Resize', r.totaltime/r.iters,'\n Transpose', t.totaltime/t.iters,'\n Inference', i.totaltime/i.iters,'\n Postprocess', p.totaltime/p.iters)
    # print(time.time()-a, (d.totaltime, d.iters), (r.totaltime,r.iters), (t.totaltime,t.iters), (i.totaltime,i.iters), (p.totaltime,p.iters))
    print('Decoder ', d.totaltime / d.iters, '\n Inference', i.totaltime / i.iters, '\n Postprocess',
          p.totaltime / p.iters)

d.kill_thread()
# r.kill_thread()
# t.kill_thread()
i.kill_thread()
p.kill_thread()
