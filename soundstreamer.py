import time,threading, os, wave, struct, math, sys, pyaudio, numpy as np, numpy.fft as fft, matplotlib.pyplot as plt
from pipeline import extract_motionbuilder_model3_3
from scipy.signal import fftconvolve
from scipy import interpolate as itp
from scipy.special import expit



class SoundStreamer():
	def __init__(self, fadeout = 1.0, volume = 0.5):
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
		data = [440.0, 350.0, 220.0, 550.0, 330.0]
		duration = 1.0
		f = np.random.choice(data)
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
		timestamp = time.time()
		self.inputbuffer.append(value)


	def resume(self):
		self.playing = True
		self.playSound()


	def start(self):
		self.playing = True


	def update(self):
		self.fadeOut()
		if self.timestamp!=None and time.time()-self.timestamp >= self.fadeout:
			self.paused = True
			self.playing = False
			self.pause(True)
			return

		while not self.pauseEvent.is_set():			
			self.playSound()
		print "playing"

		#Do interpolation here







