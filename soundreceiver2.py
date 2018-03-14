import socket
import openal
import threading
import sys
import pickle
from soundstreamer2 import *
import pygame



HOST = '127.0.0.1'   # use '' to expose to all networks
PORT = int(sys.argv[1])  # Calculation data.

class SoundReceiver():
	def __init__(self, streamer, host=HOST, port=PORT):
		self.socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self.host = host
		self.buffersize = 1024
		self.port = port
		self.streamer = streamer



	def receive(self):
		while True:
			msg = self.socket.recv(sr.buffersize)
			self.streamer.addtoStream(float(msg.split(":")[0]))

	def interpolate(self):
		while True:
			self.streamer.interpolate()

	def streamerUpdate(self):
		while True:
			self.streamer.update()
    	

	def close(self):
		self.socket.close()

	def initialize(self):
		self.socket.bind((self.host,self.port))
		self.socket.settimeout(10);





if __name__ == '__main__':
	data = []
	streamer = SoundStreamer2(HOST, PORT)	
	
	addThread = threading.Thread(target=streamer.addToStream)
	ipdThread = threading.Thread(target=streamer.interpolate)
	plyThread = threading.Thread(target=streamer.playSound)

	# msg = [440, 880, 660, 550, 330, 440, 880, 440]
	# streamer.addToStream(list(msg))
	
	while True:
		addThread.run()
		ipdThread.run()
		plyThread.run()
	
	# while True:		
	# 	for event in pygame.event.get():
	# 		print event.type
	# 		if event.type == pygame.locals.KEYDOWN:
	# 			print "key pressed"
	# 			if event.key == pygame.K_p:
	# 				streamer.pause()
	# 			if event.key == pygame.K_r:
	# 				streamer.resume()
	# 			if event.key == pygame.K_q:
	# 				pygame.quit()
	# 				sys.exit()
	# 		else:
	# 			continue
	# 	# msg = pickle.loads(sr.socket.recv(sr.buffersize))
	# 	print "message feeding in"
	# 	streamer.interpolate()
	# 	streamer.update()	
	# 	time.sleep(2)
		
		# lastSeenTime = time.time()
	streamer.end()

    