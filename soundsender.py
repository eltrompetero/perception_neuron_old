''' class for calculating and sending sound'''


import socket, time, signal, threading, sys, pickle, wave
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



def load_sound(filename):
	wav = wave.open(filename,'r')
	sig = wav.readframes('-1')
	sif = np.fromstring(sig,'int16')
	return sif



sound = load_sound("interpolate_merged_exp.wav")
datas = [440, 880, 220, 550, 330]

if __name__ == '__main__':
	wavef = wave.open("interpolate_merged_exp.wav","r")
	sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
	ind = 0
	data = wavef.readframes(1024)
	while True:
		# data = pickle.dumps(sound[ind:ind+1024])
		sock.sendto(str(np.random.choice(datas)),(HOST,PORT))
		# data = wavef.readframes(1024)
		# ind+=1024

	sock.sendto("done",(HOST,PORT))

	