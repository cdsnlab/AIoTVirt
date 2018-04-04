import subprocess


class Logger(object):
	"""docstring for Logger"""

	def __init__(self, name="", log=True, **kw):
		super().__init__(**kw)
		# self.log = log
		# if self.log:
		self.name = name
		logName = self.name + "Det_stats.log"
		self.f = open(logName, "a+")
		self.log_registry = []
		# self.width = width

	def log_cpu(self, fps, width):
		# We need to manually write linebreaks
		# self.f.write(str(width) + "\n")
		# self.f.write(str(fps) + "\n")
		log_msg = str(width) + "\n"
		log_msg += str(fps) + "\n"
		# cmd = "ps -eL -o comm,cmd,psr,pcpu | grep py >> " + logName

		# We might search for "cdsn_..:" instead of 'py' but
		# I'll leave ot like this for now"
		cmd = "ps -eL -o comm,cmd,psr,pcpu | grep py"
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
		# get output of previously executed command
		out, err = p.communicate()
		# self.f.write(out.decode())
		log_msg += out.decode()
		# Write separating empty newline
		log_msg += "\n"
		# self.f.write("\n")
		return log_msg

	def register(self, func):
		"""Each function registered should return a string that will be
		written to the log"""
		# TODO: Check that func is a callable and that it returns a string
		self.log_registry.append(func)

	def write_log(self, **kw):
		for func in self.log_registry:
			msg = func(**kw)
			self.f.write(msg)

	def close(self):
		self.f.close()
