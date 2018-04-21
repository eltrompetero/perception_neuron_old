import os, wave, struct, math, sys, numpy as np, matplotlib.pyplot as plt
from pipeline import extract_motionbuilder_model3_3
from scipy.signal import fftconvolve
from scipy import interpolate as itp
from scipy.special import expit
from openal import *
import numpy.fft as fft
import pyaudio




## Directory to save sound files
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

	


class SoundConverter():
	'''
	Converts a motion sequence to a wav file.
	'''

	def __init__(self, directory, model_name, direction, reverse_time):
		'''
			directory    : directory where sound file will be saved
			model_name   : the name of motionbuilder model
			direction    : hand direction
			reverse_time : whether the time series is going to be reversed
		'''

		# Function to be applied to raw data 
		self.function_dict = {'exp': (np.exp, 0.75), 'log': (np.log,0.5), 'sig': (expit,0.5), '': (lambda x : x, 32.0)}

		# Add any target models to this dictionary
		self.models = {'3_3': extract_motionbuilder_model3_3 }
		# Model function that would be used for loading data
		self.model = self.models[model_name]

		# Direction of hand
		self.direction = direction
		self.reverse = reverse_time

		# Loading model
		data, time = self.model(self.direction, reverse_time = self.reverse)
		self.time = data.x
		self.data = data.y
		self.directory = directory


	def save_sound(self, filename, key, duration=None):
		'''
		Interpolate the velocity and save it as soundfile.
		
			filename : Desired file name (will be saved in SOUND_DIR)
			key 	 : function that will be used to tune the input values. either exp, log, or sig
			duration : Desired duration. If unspecified, the whole motion sequence will be converted to sound.

		'''

		# Reverse the velocity
		if self.direction.lower() == 'left': 
			self.data = self.data[::-1]

		# Norm of the velocity	
		data_amp = [np.linalg.norm(x) for x in self.data]

		# Data smoothed with fft convolution
		smooth_data = fftconvolve(data_amp,np.ones(12)/12.0,mode="same")
		smooth_data = fftconvolve(smooth_data[::-1],np.ones(12)/12.0,mode="same")[::-1]

		# Create interpolation function at range (0,..., len(data)/60)
		# len(data)/60 assuming each data point is gathered at 1/60 second interval.
		data_fx = itp.interp1d(np.arange(len(smooth_data))/60.0,smooth_data)
		
		# create a saving directory if directory is not present
		if not os.path.exists(self.directory) : os.mkdir(self.directory)
		
		# prepare the function given initial function key
		function, factor = self.function_dict[key]
		print("function %s with factor %f"%(key,factor))

		# Creating wav file at sample rate of 44100Hz
		wav_samplerate = 44100.0

		# opening wav file
		wavef = wave.open(os.path.join(SOUND_DIR,'_'.join([filename,key,self.direction])+'.wav'),'w')
		wavef.setnchannels(2)
		wavef.setsampwidth(2) 
		wavef.setframerate(wav_samplerate)


		# Assuming data points are given at 60 points per second
		# Length of avatar motion should be length of data / 60 
		# If custom duration is given, use that custom duration
		max_x = len(smooth_data)/60 if duration is None else duration
		print("creating file with %s with %f seconds"%(filename,max_x))

		# Scaling the original Range 
		# ex ) np.exp(min_value*factor)*2.0*np.pi
		myMinVal = function(min(smooth_data)*factor)
		myMaxVal = function(max(smooth_data)*factor)
		myRange = myMaxVal-myMinVal

		# Minimum and maximum frequency desired
		MIN_FREQ = 150.0
		MAX_FREQ = 1000.0


		# The data is interpolated at every time step for sound sampling.
		# sound value at every time step will be the interpolated value between data points from
		# motionbuilder object.
		step = 1.0/wav_samplerate
		x_val = 0.0
		y_cum = 0

		# For debugging.
		yvals = []
		ycums = []
		yadds = []

		while x_val<max_x:
			# Get the interpolated data
			y_val  = data_fx(x_val)
			yvals.append(y_val)
			# scale from [myMinVal, myMaxVal] to [MIN_FREQ, MAX_FREQ]
			y_add  = function(y_val*factor)
			y_add  = (y_add-myMinVal)/myRange*(MAX_FREQ-MIN_FREQ)+MIN_FREQ 
			yadds.append(y_add)
			# f(t) * t
			y_cum += y_add/wav_samplerate
			ycums.append(y_cum)
			# sin(2*pi*f(t)*t)
			amp    = np.sin(y_cum*2.0*np.pi)
			# increase dt
			x_val += step
			# Use <hh because there are two channels
			wavef.writeframesraw(struct.pack('<hh',amp*32767,amp*32767))

		wavef.writeframes('')
		wavef.close()	


if __name__ == '__main__':
	converter = SoundConverter(SOUND_DIR,"3_3","Right",False)

	# Sample Usage below
	converter.save_sound("interpolate_merged","exp",3)

