import argparse
from nnet import yolo_tiny as model

def main(args):
	print 'Building the model'
	yolo = model.YOLO_tiny()
	print '\nTotal number of parameters : ', yolo.count_params()

def parser():
	"""
	Parse the arguements
	"""
	parser = argparse.ArgumentParser(description="YOLO tiny : 'Real Time Object Detection' ")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	try:
		args = parser()
		main(args)
	except Exception as E:
		print E
