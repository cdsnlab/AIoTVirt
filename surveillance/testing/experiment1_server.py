from common.streamer import TCPStreamer
from imutils.video import VideoStream
from common.logger import Logger
import asyncio
import time

# We could also use 127.0.0.1 or localhost' but in my experience, on Debian it
# works most of the times better by putting 0.0.0.0 (all loopback addresses)
ADDRESS = "0.0.0.0"
PORT = 5000

logger = Logger("streamer")

# The new logger logs all of the "registered" functions. In this case a simple
# line stating the frame number and the number of objects detected
logger.register(lambda bsent, **kw: str(bsent), logfile="bw.log")

# async def log(interval):
# 	while True:
# 		try:
# 			data = {"bsent": streamer.bytes_sent}
# 			logger.write_log(data)
# 			streamer.bytes_sent = 0
# 			await asyncio.sleep(interval)
# 		except asyncio.CancelledError:
# 			# break
# 			raise

videostream = VideoStream(usePiCamera=True, name="CameraThread").start()
time.sleep(0.1)

loop = asyncio.get_event_loop()
# Create the server based on our protocol
coro = loop.create_server(lambda: TCPStreamer(videostream),
							ADDRESS, PORT)
server = loop.run_until_complete(coro)

print('Serving on {}'.format(server.sockets[0].getsockname()))
try:
	loop.run_forever()
except KeyboardInterrupt:
	pass

server.close()
loop.run_until_complete(server.wait_closed())
loop.close()
