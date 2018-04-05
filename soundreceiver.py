import socket, pyaudio, threading, sys, pickle, pygame, time
from scipy import interpolate as itp
from streamingcontroller import *



HOST = '127.0.0.1'   # use '' to expose to all networks
PORT = int(sys.argv[1])  # Calculation data.

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
		self.socket.settimeout(10);




def load_sound(filename):
	wav = wave.open(filename,'r')
	sig = wav.readframes('-1')
	sif = np.fromstring(sig,'int16')
	return sif



def change_to_sound(duration, value, frequency):
	return np.sin(32767*np.pi*np.arange(frequency*duration)*value/frequency)




global timestamp
timestamp = None

global data
data = []

global ipdata
ipdata = []

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
sock.bind((HOST, PORT))



def interpolate():
	if data is not None and len(data) > 1 :
		itpval = data[:1]
		data_fx = itp.interp1d(np.arange(len(itpval)),itpval)
		xran = np.linspace(0,len(itpval)-1,duration*44100)
		yran = data_fx(xran)
		snd = (32767*np.cos(np.cumsum(yran)*2.0*np.pi/44100)).astype(np.int16)
		# snd = (32767.0*np.cos(np.pi*np.arange(44100*duration)*int(data[0])/44100.0)).astype(np.int16)
		del data[0]
		ipdata.append(snd)

	


def add_to_data():	
	while True:
		data.append(sock.recv(1024))
		timestamp = time.time()		




if __name__ == '__main__':	

	
	p = pyaudio.PyAudio()
	wavef = wave.open("interpolate_merged_exp.wav","r")
	sound = load_sound("interpolate_merged_exp.wav")

	
	stream = p.open(format=8,
	                channels=1,
	                rate=44100,
	                frames_per_buffer=1024,
	                output=True)


	addThread = threading.Thread(target=add_to_data)
	addThread.start()

	duration = 10#0.1
	playstamp = time.time()

	amp_fx = itp.interp1d(np.arange(2), [32767,0])
	amp_xr = np.linspace(0,1,duration*44100)
	amp_yr = amp_fx(amp_xr)
	phaseOffset = 0

	while True:
		if data is not None and len(data) > 2 :
			print "creating sound at " + str(time.time()-playstamp)
			itpval = data[:61]
			print "value : " + str(itpval)
			data_fx = itp.interp1d(np.arange(len(itpval)),itpval)
			xran = np.linspace(0,len(itpval)-1,duration*44100)
			yran = data_fx(xran)
			ycum = np.cumsum(yran)*2.0*np.pi/44100
			snd = (32767*np.cos(ycum+phaseOffset)).astype(np.int16)
			phaseOffset = ycum[-1]
			del data[:60]
			stream.write(snd)

		

    