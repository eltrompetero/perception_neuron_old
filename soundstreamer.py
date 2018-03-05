import time,threading, os, wave, struct, math, sys, pyaudio, numpy as np, numpy.fft as fft, matplotlib.pyplot as plt
from pipeline import extract_motionbuilder_model3_3
from scipy.signal import fftconvolve
from scipy import interpolate as itp
from scipy.special import expit



class SoundStreamer():
	def __init__(self, fadeout = 3.0, volume = 0.5):
		self.fadeout = fadeout
		self.p = pyaudio.PyAudio()
		self.stream = self.p.open(format=pyaudio.paFloat32,channels=2,rate=44100,output=True)

		# timestamp for fadeout
		self.timestamp = None

		# input buffer where any outside data is stored
		self.inputbuffer = []

		# buffer for data interpolation
		self.ipdbuffer = []
		self.fs = 44100


		self.playing = False
		self.paused = False
		self.volume = volume

		self.playEvent  = threading.Event()
		self.pauseEvent = threading.Event()



	def playSound(self):
		if len(self.inputbuffer)==0:
			return
		print self.inputbuffer[-1]
		data = [440.0, 350.0, 220.0, 550.0, 330.0]
		duration = 1.0
		f = self.inputbuffer[-1]
		samples = (np.sin(2*np.pi*np.arange(self.fs*duration)*f/self.fs)).astype(np.float32)
		self.stream.write(self.volume*samples)
		


	def pause(self, fadeout = True):
		if (fadeout):
			self.fadeOut()
		self.playing = False
		self.paused = True
		print "pausing sound"
		self.pauseEvent.set()


	def fadeOut(self):
		pass
		# fadeout


	def addToStream(self, value):
		# precondition : data is added one by one
		if not self.playing:
			self.playing = True
		print "adding to stream"
		self.timestamp = time.time()
		self.inputbuffer.append(value)


	def resume(self):
		self.playing = True
		self.timestamp = time.time()
		self.pauseEvent.clear()


	def init(self):
		self.playing = True


	def update(self):
		if not self.pauseEvent.is_set() and self.timestamp!=None and time.time()-self.timestamp <= self.fadeout:			
			self.playSound()
			time.sleep(0.2)
			print "playing"


		if self.pauseEvent.is_set():
			print "not playing. pauseEvent Set"
			return

		timestamp = self.timestamp if self.timestamp!=None else 0
		if not (self.timestamp!=None and time.time()-self.timestamp <= self.fadeout):
			print "not playing because data starved for " + str(time.time()-timestamp) + " seconds"
			# if self.timestamp!=None and time.time()-self.timestamp >= self.fadeout:
			# 	print "not called for more than x seconds"
			# 	self.playing = False
			# 	break
		

		#Do interpolation here







