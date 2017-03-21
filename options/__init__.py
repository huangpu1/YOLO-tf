class Options(object):
	
	def __init__(self):
		self.alpha = 0.1
		self.threshold = 0.15
		self.iou_threshold = 0.5
		self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]		