import time,threading, os, wave, struct, math, sys, pyaudio, numpy as np, numpy.fft as fft, matplotlib.pyplot as plt
import sounddevice
from pipeline import extract_motionbuilder_model3_3
from scipy.signal import fftconvolve
from scipy import interpolate as itp
from scipy.special import expit




# SoundStreamer Object
#	1. Supports interpolation of any data in the input buffer
#	2. Plays the interpolated sound
#	3. Supports pausing at data starvation


class SoundStreamer():
	def __init__(self, fadeout = 10.0, fadeout_start = 0.5, volume = 0.5):

		# For implementing fadeout
		self.fadeout = fadeout
		self.fadeout_start = fadeout_start
		self.timestamp = None
		self.playstamp = None

		# Streamer objects
		self.p = pyaudio.PyAudio()

		def callback(in_data, frame_count, time_info, status):
			datanew = self.inputbuffer[0] if len(self.inputbuffer)>0 else ""
			del inputbuffer[0]
			return (datanew, pyaudio.paContinue)

		self.stream = self.p.open(format=pyaudio.paFloat32,channels=2,rate=44100,output=True,stream_callback=callback)

		
		# input buffer where any outside data is stored
		self.inputbuffer = []
		# buffer for data interpolation
		self.ipdbuffer = []

		# sound sampling frequency
		self.frequency = 44100

		# flag for playing & pausing
		self.playing = False
		self.paused = False

		# origvolume for fadeout
		self.volume = volume
		self.origvolume = volume

		self.pauseEvent = threading.Event()
		self.duration = 2.0



	def playSound(self):		
		# print "playingSound"
		# if self.playstamp is None or (self.playstamp is not None and time.time()-self.playstamp >= 1.0):
		print self.ipdbuffer
		self.stream.write(self.ipdbuffer)		
		print "terminated"
			# self.playstamp = time.time()


	def pause(self):
		self.playing = False
		self.paused = True
		print "pausing sound"
		self.pauseEvent.set()


	def addToStream(self, value):
		# update timestamp
		self.timestamp = time.time()
		self.inputbuffer.append(value)


	def interpolate(self, key='exp'):
		if len(self.inputbuffer)<2:
			return False
		function_dict = {'exp': (np.exp, 0.75), 'log': (np.log,0.5), 'sig': (expit,0.5), '': (lambda x : x, 32.0)}		
		ipd = itp.interp1d(np.arange(len(self.inputbuffer)),self.inputbuffer)
		data1sec = np.linspace(0,len(self.inputbuffer)-1,88200)

		# np.sin(2*np.pi*np.arange(self.frequency*self.duration)*self.inputbuffer[0]/self.frequency)
		arr = np.sin(2*np.pi*np.arange(self.frequency*self.duration)*self.inputbuffer[0]/self.frequency)#ipd(data1sec)
		# print arr(np.sin(arr).astype(np.float32).tolist())
		self.ipdbuffer.append(arr)#np.append(self.ipdbuffer, arr)
		# self.stream.write(self.ipdbuffer)	# empty input buffer after interpolation.
		# self.inputbuffer = []

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
		if self.timestamp!= None and currtime-self.timestamp > self.fadeout_start:
			self.volume *= 0.5
		else:
			self.volume = self.origvolume
		if not self.pauseEvent.is_set() :
			self.playSound()
		
		if self.pauseEvent.is_set():
			if self.volume>1e-2:
				self.volume*=0.3
				self.playSound()
			else:
				print "pausing."








