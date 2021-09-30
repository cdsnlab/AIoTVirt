import cv2
cap = cv2.VideoCapture('rtsp://cdsnlab:CdsnLab7759@143.248.55.237:28557/ds-test')



w = 1200
h =  720
fps = 24
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
count = 0
max_count = 24*60*60*4

for i in range(1):
    out = cv2.VideoWriter('cam2_0913.avi', fourcc, fps, (w, h))

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame,  (w,h))
        out.write(frame)
        count+=1

        if count == max_count:
            f = open('log.txt', 'a')
            f.write(str(i)+' ' +str(count)+"\n")
            f.close()
            out.release()
            print('done')
            count=0
            break
cap.release()

f = open('log.txt', 'a')
f.write('total '+str(max_count*i)+"\n")
f.close()