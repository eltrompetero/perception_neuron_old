import socket
import openal

HOST = '127.0.0.1'   # use '' to expose to all networks
PORT = 7006  # Calculation data.

class SoundReceiver():
	def __init__(self, host=HOST, port=PORT):
		self.socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self.host = host
		self.buffersize = 1024
		self.port = port
    	

	def close(self):
		self.socket.close()

	def initialize(self):
		self.socket.bind((self.host,self.port))




if __name__ == '__main__':
	data = []
	sr = SoundReceiver()
	sr.initialize()
	while True:
		msg = sr.socket.recv(sr.buffersize)
		if 'done' in msg:
			break
		data.append(msg)

    