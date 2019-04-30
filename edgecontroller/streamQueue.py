import queue

class streamQueue:
	def __init__(self, maxSize=5):
		self.queue = queue.Queue(maxSize)
		self.maxSize = maxSize
		self.head = 0
		self.tail = 0

	def enqueue(self, data):
		if len(list(self.queue.queue)) ==self.maxSize:
			# queue is full, push one out and insert one.
			self.queue.get()
			self.queue.put(data)
		else:
			self.queue.put(data)


	def is_empty(self):
		if (len(list(self.queue.queue))):
			return False
		else:
			return True
	def flush(self):
		for i in self.maxSize:
			self.queue.get()
		print("queue emptied")