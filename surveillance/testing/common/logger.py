import subprocess
# import os


class Logger(object):
	"""docstring for Logger"""

	def __init__(self, name="", log=True, mod="w+", **kw):
		super().__init__(**kw)
		# self.log = log
		# if self.log:
		self.name = name
		self.mod = mod
		logName = self.name + ".log"
		self.f = open(logName, mod)
		self.log_registry = []
		self.line_sep = "\n"
		# self.block_sep = "\n"
		# self.width = width

	def log_cpu(self, **kw):
		# We need to manually write linebreaks
		# self.f.write(str(width) + "\n")
		# self.f.write(str(fps) + "\n")
		# log_msg = str(width) + "\n"
		# log_msg += str(fps) + "\n"
		# cmd = "ps -eL -o comm,cmd,psr,pcpu | grep py >> " + logName

		# We might search for "cdsn_..:" instead of 'py' but
		# I'll leave ot like this for now"
		cmd = "ps -eL -o comm,cmd,psr,pcpu | grep py"
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
		# get output of previously executed command
		out, err = p.communicate()
		# self.f.write(out.decode())
		log_msg = out.decode()
		# # Write separating empty newline
		# log_msg += "\n"
		# self.f.write("\n")
		return log_msg

	def register(self, func, logfile=None):
		"""Each function registered should return a string that will be
		written to the log"""
		# TODO: Check that func is a callable and that it returns a string
		if logfile is None:
			logfile = self.f
		else:
			logfile = open(logfile, self.mod)
		self.log_registry.append((func, logfile))

	def write_log(self, kw):
		for func, logfile in self.log_registry:
			msg = func(**kw)
			logfile.write(msg + self.line_sep)
		# logfile.write(self.block_sep)
	# def register(self, name, func):
	# 	"""Each function registered should return a string that will be
	# 	written to the log"""
	# 	# TODO: Check that func is a callable and that it returns a string
	# 	self.log_registry.append((name, func))

	# def write_log(self, **kw):
	# 	for name, func in self.log_registry:
	# 		msg = func(**kw)
	# 		self.f.write(msg + self.line_sep)
	# 	self.f.write(self.block_sep)

	def close(self):
		self.f.close()
