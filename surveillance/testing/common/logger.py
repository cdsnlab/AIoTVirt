from datetime import datetime
import subprocess


class Logger(object):
	"""docstring for Logger"""

	def __init__(self, width, log=False, **kw):
		super().__init__(**kw)
		self.log = log
		self.width = width

		# Open the log file for this process
		if log:
			logName = self.name + "Det_stats"
			self.f = open(logName, "a+")

	def start(self):
		self.frame_count = 0
		self.start = datetime.now()

	# def postprocessing(self):
	def approx_fps(self):
		"""Approximate fps and optionaly write ps output to log"""
		self.frame_count += 1
		time_elapsed = (datetime.now() - self.start).total_seconds()
		if time_elapsed >= 1:
			fps = self.frame_count / time_elapsed
			print("Approximate FPS: {0:.2f}".format(fps), end="\r")
			self.frame_count = 0
			self.start = datetime.now()
			# Log data if desired (one might need to manually delete
			# previously created log files)
			if self.log:
				self.write_log(fps)

	def write_log(self, fps):
		# We need to manually write linebreaks
		self.f.write(str(self.width.value) + "\n")
		self.f.write(str(fps) + "\n")
		# cmd = "ps -eL -o comm,cmd,psr,pcpu | grep py >> " + logName

		# We might search for "cdsn_..:" instead of 'py' but
		# I'll leave ot like this for now"
		cmd = "ps -eL -o comm,cmd,psr,pcpu | grep py"
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
		# get output of previously executed command
		out, err = p.communicate()
		self.f.write(out.decode())
		# Write separating empty newline
		self.f.write("\n")

	def close(self):
		if self.log:
			self.f.close()
