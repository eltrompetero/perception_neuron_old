''' class for calculating and sending sound'''


import socket, time, signal, threading, sys, pickle
import numpy as np


HOST = '127.0.0.1'   # use '' to expose to all networks
PORT = int(sys.argv[1])  # Calculation data.

class SoundSender():
	def __init__(self, host=HOST, port=PORT):
		self.socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self.host = host
		self.buffersize = 65535
		self.port = port
    	

	def close(self):
		self.socket.close()

	def initialize(self):
		self.socket.bind((self.host,self.port))





if __name__ == '__main__':
	start = time.time()
	datas = [440.0, 880.0, 220.0, 550.0, 770.0]
	ss = SoundSender()
	handler = lambda x,y : ss.close()
	signal.signal(signal.SIGINT, handler)
	while True:
		data = pickle.dumps(np.random.choice(datas,10))
		ss.socket.sendto(data,(ss.host,PORT))

	