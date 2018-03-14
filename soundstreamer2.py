import time,threading, socket, os, wave, struct, math, pickle, sys, pyaudio, numpy as np, numpy.fft as fft, matplotlib.pyplot as plt
import sounddevice
from pipeline import extract_motionbuilder_model3_3
from scipy.signal import fftconvolve
from scipy import interpolate as itp
from scipy.special import expit



class SoundStreamer2():
	def __init__(self, host, port, fadeout = 3.0, fadeout_start = 1.0, volume = 0.5):
		self.fadeout = fadeout
		self.fadeout_start = fadeout_start

		self.p = pyaudio.PyAudio()
		self.stream = self.p.open(format=pyaudio.paFloat32,channels=2,rate=44100,output=True)

		# timestamp for fadeout
		self.timestamp = None

		# input buffer where any outside data is stored
		self.inputbuffer = []
		# buffer for data interpolation
		self.ipdbuffer = np.array([])

		# default duration and wave sampling frequency
		self.duration = 8.0
		self.frequency = 44100

		# flag for playing & pausing
		self.playing = False
		self.paused = False

		self.volume = volume
		self.origvolume = volume

		self.playEvent  = threading.Event()
		self.pauseEvent = threading.Event()
		self.recvPauseEvent = threading.Event()

		# socket for receiving data
		self.socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self.socket.bind((host,port))
		self.socket.settimeout(10);



	def playSound(self):		
		while not self.pauseEvent.is_set():
			print "playing"
			samples1 = np.array(self.ipdbuffer[:len(self.ipdbuffer)]).astype(np.float32)
			print samples1
			print "sample len : " + str(len(samples1))
			self.stream.write(self.volume*samples1)
		# if len(self.inputbuffer) > 100:
		# 	del self.inputbuffer[:8]
		


	def pause(self):
		self.playing = False
		self.paused = True
		print "pausing sound"
		self.pauseEvent.set()


	def pauseRecv(self):
		self.recvPauseEvent.set()


	def addToStream(self):
		while not self.recvPauseEvent.is_set():
			print "adding"
			recvdata = pickle.loads(self.socket.recv(1024))
		# precondition : data is added one by one
			prevtimestamp = self.timestamp
			self.timestamp = time.time()
			self.inputbuffer+=list(recvdata)


	def interpolate(self, key='exp'):
		function_dict = {'exp': (np.exp, 0.75), 'log': (np.log,0.5), 'sig': (expit,0.5), '': (lambda x : x, 32.0)}

		while not self.pauseEvent.is_set():
			datas = self.inputbuffer[:10]
			
			# if len(self.ipdbuffer) > self.frequency:
			# 	break
			for data in datas:
				self.ipdbuffer = np.append(self.ipdbuffer,((np.sin(2*np.pi*np.arange(self.frequency*self.duration)*data/self.frequency)).astype(np.float32)))
			
			print "buffer len : " + str(len(self.ipdbuffer))
			del self.inputbuffer[:10]

		# smooth_data = fftconvolve(self.inputbuffer,np.ones(12)/12.0)
		# smooth_data = fftconvolve(smooth_data[::-1],np.ones(12)/12.0)[::-1]
		# # if self.direction.lower() == 'left': 
		# # 	smooth_data = smooth_data[::-1]

		# print len(smooth_data)
		# data_fx = itp.interp1d(np.arange(len(smooth_data))/60.0,smooth_data)			

		# function, factor = function_dict[key]
		# print("function %s with factor %f"%(key,factor))

		# wav_samplerate = self.frequency
		# max_x = max(len(smooth_data)/60, 1.0)

		# myMinVal = function(min(smooth_data)*factor)
		# myMaxVal = function(max(smooth_data)*factor)
		# myRange = myMaxVal-myMinVal


		# MIN_FREQ = 150.0
		# MAX_FREQ = 1000.0

		# step = 1.0/wav_samplerate
		# x_val = 0.0
		# y_cum = 0

		# ind = 0

		# yvals = []
		# ycums = []
		# yadds = []

		# while x_val<max_x:
		# 	# print x_val
		# 	y_val  = data_fx(x_val)
		# 	y_add  = function(y_val*factor) #rescaled and transformed
		# 	y_add  = (y_add-myMinVal)/myRange*(MAX_FREQ-MIN_FREQ)+MIN_FREQ # scale from one range to another
		# 	y_cum += y_add/wav_samplerate # f(t)*t
		# 	amp    = np.sin(y_cum*2.0*np.pi)
		# 	x_val += step
		# 	ind   += 1		
		# 	self.ipdbuffer.append(amp)

		return True


	def resume(self):
		self.playing = True
		self.timestamp = time.time()
		self.volume = self.origvolume
		self.paused = False		
		self.pauseEvent.clear()


	def init(self):
		self.playing = True


	def update(self):
		currtime = time.time()
		if not self.pauseEvent.is_set() and self.timestamp!=None and currtime-self.timestamp <= self.fadeout:			
			self.playSound()
			if currtime-self.timestamp > self.fadeout_start:
				self.volume *= 0.5
			else:
				self.volume = self.origvolume


		if self.pauseEvent.is_set():
			if self.volume>1e-2:
				print "paused but volume : " + str(self.volume)
				self.volume*=0.3
				self.playSound()

			print "not playing. pauseEvent Set"
			return

		timestamp = self.timestamp if self.timestamp!=None else 0
		if not (self.timestamp!=None and currtime-self.timestamp <= self.fadeout):
			print "not playing because data starved for " + str(currtime-timestamp) + " seconds"
			# if self.timestamp!=None and time.time()-self.timestamp >= self.fadeout:
			# 	print "not called for more than x seconds"
			# 	self.playing = False
			# 	break
		

		#Do interpolation here







