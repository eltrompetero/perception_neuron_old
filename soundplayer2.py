import os, wave, struct, math, sys, numpy as np, matplotlib.pyplot as plt
from pipeline import extract_motionbuilder_model3_3
from scipy.signal import fftconvolve
from scipy import interpolate as itp
from scipy.special import expit
from openal import *
import numpy.fft as fft
import pyaudio



SOUND_DIR = "soundfiles_leftright"


def load_sound(filename):
	wav = wave.open(filename,'r')
	sig = wav.readframes('-1')
	sif = np.fromstring(sig,'int16')
	return sif


def save_sound(filename, freqlist, exp_samplerate, wav_samplerate=44100.0):
	'''
	Save frequency list into a wav sound file
	'''
	duration = 1.0/exp_samplerate # 1/samplerate seconds per each sample
	wavef = wave.open(filename,'w')
	wavef.setnchannels(1)
	wavef.setsampwidth(2) 
	wavef.setframerate(wav_samplerate)
	for frequency in freqlist:
		for i in range(int(duration * wav_samplerate)):
			value = np.int16(32767.0*math.cos(frequency*math.pi*float(i)/float(wav_samplerate))) # converting into PCM format
			data = struct.pack('<h', value)
			wavef.writeframesraw( data )
	wavef.writeframes('')
	wavef.close()




def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

	


class SoundPlayer():
	def __init__(self, directory, model_name, direction, reverse_time):
		self.models = {'3_3': extract_motionbuilder_model3_3 }
		self.model = self.models[model_name]
		self.function_dict = {'exp': (np.exp, 0.75), 'log': (np.log,0.5), 'sig': (expit,0.5), '': (lambda x : x, 32.0)}
		self.direction = direction
		self.reverse = reverse_time
		data, time = self.model(self.direction, reverse_time = self.reverse)
		self.time = data.x
		self.data = data.y
		self.directory = directory



	def save_sound_leftright(self, filename, key, duration=None):
		positions = self.data
		moveRight = [dp[0] > 0 for dp in positions for x in range(44100/60)]


		data_amp = [np.linalg.norm(x) for x in self.data]
		data_x = [v[1] for v in self.data]
		data_exp = np.exp(np.abs(data_x*10))
		max_dx = max(np.exp(np.abs(data_x*10)))

		smooth_data = fftconvolve(data_amp,np.ones(12)/12.0,mode="same")
		smooth_data = fftconvolve(smooth_data[::-1],np.ones(12)/12.0,mode="same")[::-1]

		smooth_data_x = fftconvolve(data_x,np.ones(12)/12.0,mode="same")
		smooth_data_x = fftconvolve(smooth_data_x[::-1],np.ones(12)/12.0,mode="same")[::-1]


		if self.direction.lower() == 'left': 
			smooth_data = smooth_data[::-1]
			moveRight = leftright[::-1]
			smooth_data_x = smooth_data_x[::-1]


		# Interpolation
		data_fx = itp.interp1d(np.arange(len(smooth_data))/60.0,smooth_data)
		data_fx_x = itp.interp1d(np.arange(len(smooth_data_x))/60.0,smooth_data_x)
		

		if not os.path.exists(self.directory) : os.mkdir(self.directory)
		
	
		function, factor = self.function_dict[key]
		print("function %s with factor %f"%(key,factor))

		wav_samplerate = 44100.0

		# opening wav file
		wavef = wave.open(os.path.join(SOUND_DIR,'_'.join([filename,key,self.direction])+'.wav'),'w')
		wavef.setnchannels(2)
		wavef.setsampwidth(2) 
		wavef.setframerate(wav_samplerate)

		duration = len(smooth_data)/60 if duration is None else duration
		print("creating file with %f seconds"%duration)

		# Scaling the original Range 
		# ex ) np.exp(min_value*factor)*2.0*np.pi
		MIN_FREQ = 150.0
		MAX_FREQ = 1000.0

		t = np.arange(int(wav_samplerate*duration))/wav_samplerate
		rawVelocity = data_fx(t)
		xVelocity = data_fx_x(t)
		freq=function(rawVelocity*factor)

		factor_left = np.exp(-np.sign(xVelocity)*np.abs(xVelocity*10))
		factor_right = np.exp(np.sign(xVelocity)*np.abs(xVelocity*10))
		max_xvel = max(max(factor_left),max(factor_right))


		freq=(freq-freq.min())/(freq.max()-freq.min())*(MAX_FREQ-MIN_FREQ)+MIN_FREQ
		phase=np.cumsum(freq)/wav_samplerate
		amp    = np.sin(phase*2.0*np.pi)

		factor = 10**int(np.log(max_xvel)/np.log(10))/2
		print factor
		for i, a in enumerate(amp):
			fl = min(max(0.1,factor*factor_left[i]/max_xvel),1.0)
			fr = min(max(0.1,factor*factor_right[i]/max_xvel),1.0)
			wavef.writeframesraw(struct.pack('<hh',a*fl*32767,a*fr*32767))



	def save_sound(self, filename, key, duration=None):
		data_amp = [np.linalg.norm(x) for x in self.data]
		smooth_data = fftconvolve(data_amp,np.ones(12)/12.0,mode=same)
		smooth_data = fftconvolve(smooth_data[::-1],np.ones(12)/12.0,mode=same)[::-1]

		if self.direction.lower() == 'left': 
			smooth_data = smooth_data[::-1]


		# Interpolation
		data_fx = itp.interp1d(np.arange(len(smooth_data))/60.0,smooth_data)
		

		if not os.path.exists(self.directory) : os.mkdir(self.directory)
		

	
		function, factor = self.function_dict[key]
		print("function %s with factor %f"%(key,factor))


		wav_samplerate = 44100.0

		# opening wav file
		wavef = wave.open(os.path.join(SOUND_DIR,'_'.join([filename,key,self.direction])+'.wav'),'w')
		wavef.setnchannels(2)
		wavef.setsampwidth(2) 
		wavef.setframerate(wav_samplerate)

		max_x = len(smooth_data)/60 if duration is None else duration
		print("creating file with %f seconds"%max_x)

		# Scaling the original Range 
		# ex ) np.exp(min_value*factor)*2.0*np.pi

		myMinVal = function(min(smooth_data)*factor)
		myMaxVal = function(max(smooth_data)*factor)
		myRange = myMaxVal-myMinVal


		MIN_FREQ = 150.0
		MAX_FREQ = 1000.0

		step = 1.0/wav_samplerate
		x_val = 0.0
		y_cum = 0

		ind = 0

		yvals = []
		ycums = []
		yadds = []

		while x_val<max_x:
			y_val  = data_fx(x_val)
			yvals.append(y_val)
			y_add  = function(y_val*factor) #rescaled and transformed
			y_add  = (y_add-myMinVal)/myRange*(MAX_FREQ-MIN_FREQ)+MIN_FREQ # scale from one range to another
			yadds.append(y_add)
			y_cum += y_add/wav_samplerate # f(t)*t
			ycums.append(y_cum)
			amp    = np.sin(y_cum*2.0*np.pi)
			x_val += step
			wavef.writeframesraw(struct.pack('<h',amp*32767))

		wavef.writeframes('')
		wavef.close()	


if __name__ == '__main__':
	sp = SoundPlayer(SOUND_DIR,"3_3","Right",False)
	sp.save_sound_leftright("interpolate_leftright","exp")

