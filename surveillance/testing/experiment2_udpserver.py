# from IPython.terminal.embed import InteractiveShellEmbed
# import numpy as np
import asyncio
import struct
# import cv2
# import io


class UDPClient(asyncio.DatagramProtocol):
	"""Serve frames to the clients"""

	# asyncio already implements the generation of sockets, receiving etc, we
	# just have to provide the functions to react to certain events
	def __init__(self, verbose):
		self.bytes_sent = 0
		self.verbose = True
		self.buff = b""
		self.dtypes = {0: "IMG", 1: "TEXT"}
		# self.method = method

	def connection_made(self, transport):
		# Print information to easily see if connections are arriving
		print("conn made")
		if self.verbose:
			name = transport.get_extra_info("peername")
			print("Incoming connection from host {0}".format(name))
		# The transport is basically the socket object wrapped inside asyncio
		self.transport = transport

	def datagram_received(self, data, addr):
		# print("data recvd")
		# For more complicated scenarios we can have a buffer per ip address
		# (assuming that the ip ddresses won't change and therefore is a good
		# identifier for the images). We could create a dictionary/hash table
		# dynamically and store the images there.
		try:
			# The client sends msgs in the form "TYPE,<data>" We know the format
			# of the packet but it can be of variable length so we make it a
			# variable.
			# B: uchar (1B), I: int (4B) whichi is why we substract 5 from the
			# length.
			if len(data) > 3:
				decode_len = len(data) - 5
				dtype, seq, data = struct.unpack("<BI{0}s".format(decode_len), data)
				if self.dtypes[dtype] == "IMG":
					self.buff += data
			elif len(data) == 3 and data.decode() == "END":
				# print("Image received")
				print("buff_len: {}".format(len(self.buff)))
				# data = np.fromstring(self.buff, dtype="uint8")
				# frame = cv2.imdecode(data, 1)
				# ipshell = InteractiveShellEmbed(banner1='Dropping into IPython',
				# 								exit_msg='Leaving Interpreter, back to program.')
				# ipshell('***Called from top level. '
				# 'Hit Ctrl-D to exit interpreter and continue program.\n'
				# 'Note that if you use %kill_embedded, you can fully deactivate\n'
				# 'This embedded instance so it will never turn on again')
				# cv2.imshow("img", frame)
				# # If the 'q' key is pressed, stop the loop
				# cv2.waitKey(1)
				# print(addr)
				self.buff = b""
				self.transport.sendto("ACK".encode(), addr)
			else:
				print("Didn't receive data correctly...")
				print(data)
		except Exception as e:
			print("Error while processing received request")
			print(data)
			raise e

	def reset_bytes_count(self):
		self.bytes_sent = 0


ADDRESS = "0.0.0.0"
PORT = 5000

loop = asyncio.get_event_loop()
# Create the server based on our protocol
coro = loop.create_datagram_endpoint(lambda: UDPClient(True), local_addr=(ADDRESS, PORT))
transport, protocol = loop.run_until_complete(coro)

print('Serving on {}'.format((ADDRESS, PORT)))
try:
	loop.run_forever()
except KeyboardInterrupt:
	pass

# server.close()
transport.close()
loop.close()
