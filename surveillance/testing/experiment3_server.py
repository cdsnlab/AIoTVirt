# from IPython.terminal.embed import InteractiveShellEmbed
from contextlib import suppress
import pandas as pd
import argparse
import logging
import asyncio
import struct


class UDPClient(asyncio.DatagramProtocol):
	"""Serve frames to the clients"""

	# asyncio already implements the generation of sockets, receiving etc, we
	# just have to provide the functions to react to certain events
	def __init__(self, queue, stop_flag, restart_flag):
		self.bytes_sent = 0
		# Not needed here
		# self.verbose = True
		self.stop_flag = stop_flag
		self.restart_flag = restart_flag
		self.buff = b""
		self.pck_cnt = 0
		self.img_cnt = 0
		self.stop_pck_cnt = None
		self.dtypes = {0: "IMG", 1: "TEXT"}
		self.queue = queue

	def connection_made(self, transport):
		# Print information to easily see if connections are arriving
		logging.info("UDP Listener started")
		# The transport is basically the socket object wrapped inside asyncio
		self.transport = transport
		asyncio.ensure_future(self.client_restart())

	def datagram_received(self, data, addr):
		# logging.debug("UDP Listener received data")
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
				logging.debug("UDPListener data len: {}".format(len(data)))
				dtype, seq, data = struct.unpack("<BI{0}s".format(decode_len), data)
				if self.dtypes[dtype] == "IMG":
					self.buff += data
					self.pck_cnt += 1
			elif len(data) == 3 and data.decode() == "END":
				# logging.info("Image received")
				logging.debug("buff_len: {}".format(len(self.buff)))
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

				# Increase the counter for received images (currently not used).
				# Also empty the buffer
				self.img_cnt += 1
				self.buff = b""
				# self.transport.sendto("ACK".encode(), addr)
				if self.img_cnt == 5:
					# Remember how many packages were received until the
					# theoretical stop point
					self.stop_pck_cnt = self.pck_cnt
					# Inform the ConnManager he can send the stop message to the
					# client
					self.stop_flag.set()
			else:
				logging.warning("Didn't receive data correctly...")
				print(data)
		except Exception as e:
			logging.error("Error while processing received request")
			print(data)
			raise e

	async def client_restart(self):
		try:
			while True:
				# The ConnManager will tell us if we should restart
				await self.restart_flag.wait()
				logging.debug("UDPListener: clearing counters")
				self.restart_flag.clear()
				pck_overdue = self.pck_cnt - self.stop_pck_cnt
				self.pck_cnt = 0
				self.stop_pck_cnt = 0
				self.img_cnt = 0
				self.buff = b""
				# Send to the conn manager the number of overdue packages we
				# received
				await self.queue.put(pck_overdue)
		except asyncio.CancelledError:
			raise

	def reset_bytes_count(self):
		self.bytes_sent = 0


class ConnManager(asyncio.Protocol):
	"""docstring for ConnManager"""

	def __init__(self, queue, stop_flag, client_restart, num_measurements):
		self.queue = queue
		self.client_restart = client_restart
		self.stop_flag = stop_flag
		self.num_measurements = num_measurements
		self._meas_complete = asyncio.Event()

	def connection_made(self, transport):
		name = transport.get_extra_info("peername")
		self.transport = transport
		logging.info("Incoming connection from host {0}".format(name))
		# self.transport.write("start".encode())
		asyncio.ensure_future(self._send_stop())
		asyncio.ensure_future(self.measure())

	def data_received(self, data):
		# Once the client confirms he has received the stop message the measure
		# is complete
		logging.debug("ConnManager: received data '{}'".format(data))
		data = data.decode()
		if data.upper() == "STOP ACK":
			logging.debug("ConnManager: Stop ACK received")
			self._meas_complete.set()

	async def _send_stop(self):
		try:
			while True:
				# Wait for the UDPListener to count 5 images packages
				await self.stop_flag.wait()
				self.stop_flag.clear()
				logging.debug("Stop flag set")
				# Tell the client he should _stop_ sending frames
				# print(self.transport)
				self.transport.write("stop".encode())
		except asyncio.CancelledError:
			raise

	async def measure(self):
		t_measurements = []
		for data in ["full", "masked", "cropped"]:
			# self.dtype = data
			col = []
			for i in range(self.num_measurements):
				# Tell the client he should _start_ sending frames
				cmd = "start," + data
				self.transport.write(cmd.encode())
				# Wait until the measurement is complete
				await self._meas_complete.wait()
				logging.debug("ConnManager: Measure completed")
				self._meas_complete.clear()
				# Inform the UDPListener it should restart its state
				self.client_restart.set()
				pck_overdue = await self.queue.get()
				logging.debug("ConnManager: recvd from queue '{}'".format(pck_overdue))
				# Append measurements
				col.append(pck_overdue)
			t_measurements.append(col)
		d = {"t_full": t_measurements[0], "t_masked": t_measurements[1], "t_cropped": t_measurements[2]}
		df = pd.DataFrame.from_dict(d)
		df.to_csv("experiment3.csv")
		logging.info("Measurements finished and saved")
		logging.warning("This is the end of the program but I don't know how to exit cleanly... Please pres ctrl c ;)")
		# raise KeyboardInterrupt


parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
parser.add_argument("--num-measurements", "-n", default=50, type=int)
args = parser.parse_args()

if args.debug:
	level = logging.DEBUG
else:
	level = logging.INFO
logging.basicConfig(level=level)


ADDRESS = "0.0.0.0"
PORT = 5000

loop = asyncio.get_event_loop()
queue = asyncio.Queue(loop=loop)
stop_flag = asyncio.Event()
restart_flag = asyncio.Event()
# Create the server based on our protocol
udp_coro = loop.create_datagram_endpoint(lambda: UDPClient(queue, stop_flag=stop_flag,
										restart_flag=restart_flag), local_addr=(ADDRESS, PORT))
tcp_coro = loop.create_server(lambda: ConnManager(queue, stop_flag=stop_flag,
									client_restart=restart_flag,
									num_measurements=args.num_measurements), ADDRESS, PORT)
udp_server = loop.run_until_complete(udp_coro)
tcp_server = loop.run_until_complete(tcp_coro)

print('Serving on {}'.format((ADDRESS, PORT)))
try:
	loop.run_forever()
except KeyboardInterrupt:
	pass

udp_server[0].close()
tcp_server.close()
pending = asyncio.Task.all_tasks()
for task in pending:
	task.cancel()
	# Now we should await task to execute it's cancellation.
	# Cancelled tasks raise asyncio.CancelledError that we can suppress:
	with suppress(asyncio.CancelledError):
		loop.run_until_complete(task)
loop.close()
