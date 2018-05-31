from common.detector import CascadeDetector
from common.detector import HOGDetector
from datetime import datetime
from math import ceil
import pandas as pd
import numpy as np
import argparse
import imutils
import socket
import struct
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--img", "-i", default="/media/dv/784A74A44A746134/Users/Nicho/Pictures/Starnberg.jpg")
parser.add_argument("--descriptor", help="path to cascade descriptor",
					default="../resources/cascades/haarcascade_frontalcatface.xml", type=str)
parser.add_argument("--detector", choices=["hog", "cascade"], default="cascade", help="type of detector")
parser.add_argument("--scale", "-s", help="Scale Factor", default=1.1, type=float)
parser.add_argument("--winstride", "-w", help="Window Stride", default=(4, 4), type=tuple)
parser.add_argument("--display", "-d", action="store_true",
					help="Set flag to display frame on screen")
parser.add_argument("--local-port", default=3000)
parser.add_argument("--host", help="DNS hostname", type=str, default="raspberrypi2")
parser.add_argument("--ip", help="Destination IP ",
					default="0.0.0.0")
parser.add_argument("-p",
					"--dest-port",
					help="Destination port to send the data to. "
						"(Default: 5000)",
					default=5000)
args = parser.parse_args()

DATAGRAM_PAYLOAD_SIZE = 512
dtypes = {"IMG": 0, "TEXT": 1}


def send_img(payload):
	# Calculate how many datagrams we will need to send the image
	num_datagrams = ceil(len(payload) / DATAGRAM_PAYLOAD_SIZE)
	print("num datagrams: {}".format(num_datagrams))
	dtype = dtypes["IMG"]
	# The last datagram will most probably be smaller than the rest so send it
	# separately
	for seq in range(num_datagrams - 1):
		start_idx = seq * DATAGRAM_PAYLOAD_SIZE
		end_idx = (seq + 1) * DATAGRAM_PAYLOAD_SIZE
		fmt = "<BI{}s".format(DATAGRAM_PAYLOAD_SIZE)
		dgram_payload = struct.pack(fmt, dtype, seq, payload[start_idx:end_idx])
		soc.sendto(dgram_payload, (args.ip, args.dest_port))
	seq = num_datagrams - 1
	start_idx = seq * DATAGRAM_PAYLOAD_SIZE
	dgram_payload = struct.pack("<BI{0}s".format(len(payload)), dtype, seq, payload[start_idx:])
	soc.sendto(dgram_payload, (args.ip, args.dest_port))
	soc.sendto("END".encode(), (args.ip, args.dest_port))


soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
soc.settimeout(4)
# try:
# 	soc.bind(('0.0.0.0', args.local_port))
# except Exception as e:
# 	print(e)
# 	print("WARNING: failed to bind to port number: {}.\
# 			Random port will be selected by OS later.".format(args.local_port))

if args.detector == "hog":
	detector = HOGDetector(scaleFactor=args.scale, winStride=args.winstride)
elif args.detector == "cascade":
	detector = CascadeDetector(args.descriptor)

try:
	# while run_loop:
	# Get a frame from camera thread
	# t0 = datetime.now()
	frame = cv2.imread(args.img)
	frame = imutils.resize(frame, width=300)
	# t1 = datetime.now()
	# print("Req time: {0}".format((t1 - t0).total_seconds()))
	if frame is None:
		pass
		# break
	# frame = imutils.resize(frame, width=width)
	detector.frame = frame
	# t2 = datetime.now()
	# print("Alloc Time: {0}".format((t2 - t1).total_seconds()))
	# Break if frammed couldn'tbe grabbed
	# Get the box corners for the detected objects
	objRects = detector.detect()
	# t3 = datetime.now()
	# print("Obj Det time: {0}".format((t3 - t2).total_seconds()))
	# If desired, display the frame
	if args.display:
		frameClone = frame.copy()
		for (fX, fY, fW, fH) in objRects:
			cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

		# Show our detected objects, then clear the frame in
		# preparation for the next frame
		# cv2.imshow("img", frameClone)

		# # If the 'q' key is pressed, stop the loop
		# cv2.waitKey(0)
	img_data = cv2.imencode(".jpg", frame,
							[int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tostring()
	print("Uncompressed size: {0}".format(len(img_data)))

	# I had problems when creating an image using np.zeros. With zeros_like it
	# works as expected so I assume the error was due to data type
	frameClone = np.zeros_like(frame)
	t0 = datetime.now()
	# Copy from the incoming frames the parts that we are interested in. The
	# copy time was in the order of magnitude of 100 micro seconds so I think it
	# wouldn't really affect performance since we are aiming at ~10fps which
	# implies 100ms
	for (fX, fY, fW, fH) in objRects:
		cropped = frame[fY: fY + fH, fX:fX + fW]
		frameClone[fY: fY + fH, fX:fX + fW] = cropped
	# cropped = frame[fY: fY + fH, fX:fX + fW]
	ellapsed = datetime.now() - t0
	print("Time for copying ROI: {0}".format(ellapsed.total_seconds()))
	# cv2.imshow("img", frameClone)
	# # If the 'q' key is pressed, stop the loop
	# cv2.waitKey(0)
	# Compress the images
	masked_data = cv2.imencode(".jpg", frameClone,
							[int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tostring()
	print("Masked size: {0}".format(len(masked_data)))
	cropped_data = cv2.imencode(".jpg", cropped,
							[int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tostring()
	print("Compressed size: {0}".format(len(cropped_data)))
	t_measurements = []
	for data in [img_data, masked_data, cropped_data]:
		col = []
		for i in range(50):
			t0 = datetime.now()
			send_img(data)
			res = soc.recv(32)
			if res.decode() == "ACK":
				t_send = datetime.now() - t0
				print("send time: {t}".format(t=t_send.total_seconds()))
				col.append(t_send)
			else:
				print("Didn't receive ACK")
		t_measurements.append(col)
		# send_img(masked_data)
		# send_img(cropped_data)

	d = {"t_full": t_measurements[0], "t_masked": t_measurements[1], "t_cropped": t_measurements[2]}
	df = pd.DataFrame(data=d)
	df.to_csv("/home/dv/Documents/Uni/KAIST/CDSN/Chameleon/surveillance/t_server_measurements")

except KeyboardInterrupt:
	print("\nClosing application")
	soc.close()
except Exception as e:
	raise e
