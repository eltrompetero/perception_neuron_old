''' class for calculating and sending sound'''


import socket


HOST = '127.0.0.1'   # use '' to expose to all networks
PORT = 7006  # Calculation data.

class SoundSender():
	def __init__(self, host=HOST, port=PORT):
		self.socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self.host = HOST
		self.buffersize = 1024
		self.port = port
    	

	def close(self):
		self.socket.close()

	def initialize(self):
		self.socket.bind((self.host,self.port))




if __name__ == '__main__':
	datas = [440 for x in range(10)]
	ss = SoundSender()
	for data in datas:
		ss.socket.sendto(str(data),(ss.host,PORT))

	ss.socket.sendto('done',(ss.host,PORT))