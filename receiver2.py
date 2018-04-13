import socket, pyaudio, threading, sys, pickle, pygame, time
from scipy import interpolate as itp
from streamingcontroller import *
import array



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

# data buffer where frequency is received
global data
data = []


sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
sock.bind((HOST, PORT))



def add_to_data():	
	while True:
		data.append(sock.recv(1024))
		timestamp = time.time()		



def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == '__main__':	

	duration = 3 if len(sys.argv) <3 else int(sys.argv[2])	
	phaseOffset = 0

	p = pyaudio.PyAudio()
	wavef = wave.open("interpolate_merged_exp.wav","r")
	sound = load_sound("interpolate_merged_exp.wav")

	global queue
	queue = []

	def callback(in_data, frame_count, time_info, status):
		dd = queue.pop() if len(queue)>0 else ''.join(map(lambda x : struct.pack('<h',x), np.array([0 for i in range(44100*duration)])))
		return (dd, pyaudio.paContinue)

	
	
	stream = p.open(format=8,
	                channels=1,
	                rate=44100,
	                frames_per_buffer=44100*duration, # How many frames are you going to play per buffer?
	                output=True,
	                stream_callback=callback)


	# Start collecting data
	addThread = threading.Thread(target=add_to_data)
	addThread.start()

	
	playstamp = time.time()
	stream.start_stream()

	yprev =[]
	ycurr = []
	snd = []
	while True:
		if data is not None and len(data) > 2 :
			print "creating sound at " + str(time.time()-playstamp)
			itpval = data[:60]
			x = np.linspace(0,len(itpval),len(itpval))
			y = itpval

			data_fx = itp.interp1d(x,y,kind='cubic')
			xran = np.linspace(0,len(itpval),duration*44100)
			yran = data_fx(xran)
			ycum = np.cumsum(yran)*2.0*np.pi/44100

			if snd!=[]:
				yprev = snd[-500:]


			snd = (32767*np.cos(ycum+phaseOffset)).astype(np.int16)
			ycurr = snd[:500]
			phaseOffset += ycum[-2]
			del data[:59]

			# if yprev!=[] and ycurr!=[]:
			# 	plt.plot(yprev)
			# 	plt.plot(range(500,1000),ycurr[:500])
			# 	plt.show()
			
			val = ''.join(map(lambda x : struct.pack('<h',x), snd))
			
			queue+=(list(chunks(val,44100*duration*2)))

	stream.stop_stream()
	addThread._stop_event.set()


		

    