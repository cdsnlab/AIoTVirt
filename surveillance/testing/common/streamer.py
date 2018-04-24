from numpy import savez_compressed
import asyncio
import imutils
import cv2
import io


class TCPStreamer(asyncio.Protocol):
	"""Serve frames to the clients"""

	# asyncio already implements the generation of sockets, receiving etc, we
	# just have to provide the functions to react to certain events
	def __init__(self, videostream):
		self.videostream = videostream
		self.bytes_sent = 0
		# self.method = method

	def connection_made(self, transport):
		# Print information to easily see if connections are arriving
		name = transport.get_extra_info("peername")
		print("Incoming connection from host {0}".format(name))
		# The transport is basically the socket object wrapped inside asyncio
		self.transport = transport

	def data_received(self, data):
		try:
			# The client sends msgs in the form "GET,<frame_width>"
			msg, size, method = data.decode().split(",")
			msg = msg.upper()
			size = int(size)
			# Sanity check and maybe we cann add other msgs if needed
			if msg == "GET":
				# Grab frame and send it to the client
				frame = self.videostream.read()
				frame = imutils.resize(frame, width=size)
				# Instantiate the buffer
				file = io.BytesIO()
				if method == "np_compress":
					# Save Zipped numpy array to the buffer. The function takes
					# the arrays to be saved as kwargs, in this case we are
					# saving the image in the variable _frame_ under the kwarg
					# "frame"
					savez_compressed(file, frame=frame)
					# Place pointer at the beginning of the buffer
					file.seek(0)
					out = file.read()
					# Send the frame followed by the "END" to signalize the end
					# of the transmission
					self.transport.write(out)
					self.transport.write("END".encode())
				elif method == "jpg_compress":
					# Quality 95 is actually the std value but I put it there
					# just to be able to play around with the parameter in case
					# it's needed.
					img_data = cv2.imencode(".jpg", frame,
											[int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tostring()
					buff = io.BytesIO(img_data)
					out = buff.read()
					self.transport.write(out)
					self.transport.write("END".encode())
					self.bytes_sent += len(out)
				else:
					pass
		except Exception as e:
			print("Error while processing received request")
			print(data)
			raise e

	def reset_bytes_count(self):
		self.bytes_sent = 0
