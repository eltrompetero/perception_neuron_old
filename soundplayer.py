import os, wave, struct, math, sys, numpy as np, matplotlib.pyplot as plt
from pipeline import extract_motionbuilder_model3_3
from scipy.signal import fftconvolve
from scipy import interpolate as itp
from scipy.special import expit
import numpy.fft as fft


SOUND_DIR = "soundfiles_short"


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



def save_figure(data_x, data_y, xlabel, ylabel, title, filename):
	plt.plot(data_x, data_y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(filename)
	plt.close()



def bucket_data(data, bucket_len, precision=2):
	'''
	categorizes frequency into N buckets
	'''
	min_data = min(data)
	max_data = max(data)
	bucket_size = (max_data-min_data)*1.0/bucket_len
	prec_mul = 10**precision
	buckets = [x*1.0/prec_mul for x in range(int(min_data*prec_mul),int(max_data*prec_mul),int(bucket_size*prec_mul))]
	buckets.append(buckets[-1]+round(bucket_size,precision))
	bucket_pair = list(enumerate(zip(buckets[:-1],buckets[1:])))	
	new_data = []
	for point in data:
		ind = filter(lambda x : point>x[1][0] and point<=x[1][1], bucket_pair)[0][0]
		new_data.append(ind)
	new_data = [x+1 if x<bucket_len else x for x in new_data]
	return new_data



def convert_to_frequency(data, range=False, minf=240, maxf=720):
	medf = (minf+maxf)/2
	interval = (maxf-minf)/max(data)
	median = max(data)/2+1 if max(data)%2==0 else max(data)/2+0.5
	freq = [medf+(median-x)*interval for x in data]
	return freq



def main():
	v,t = extract_motionbuilder_model3_3('Right',reverse_time=False)
	time = v.x
	data = v.y
	data_amp = [np.linalg.norm(x) for x in data]

	# save_figure(time,data_amp,"time (sec)","velocity magnitude","velocity magnitude value","../data-magnitude.png")
	bucket_sizes = [3, 4, 5, 10, 15, 20]

	# for bucket in bucket_sizes:
		# data_bucketed = bucket_data(data_amp,bucket)
		# freqlist = convert_to_frequency(data_bucketed)
	freqlist = [x*1200 for x in data_amp]
	# save_figure(time,data_bucketed,"reverse_time (sec)","bucket label","bucketed data size %d"%bucket,
	# 	"../data-bucketed-size%d.png"%bucket)
	save_sound("../righthand_rawinput.wav",freqlist,60)



def moving_average(data, window_radius = 1):
	result = data
	change_points = filter(lambda i : data[i]!=data[i+1], range(len(data)-1))
	frequencypair = map(lambda x : (data[x],data[x+1]), change_points)
	for (index, (prev,next)) in zip(change_points,frequencypair):
		result[index:index+window_radius+1] = [(prev+next)/2 for i in range(len(result[index:index+window_radius+1]))]
	return result




def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

	


if __name__ == '__main__':
	v,t = extract_motionbuilder_model3_3('Right',reverse_time=False)
	time = v.x
	data = v.y
	data_amp = [np.linalg.norm(x) for x in data]
	smooth_data = fftconvolve(data_amp,np.ones(12)/12.0)
	smooth_data = fftconvolve(smooth_data[::-1],np.ones(12)/12.0)[::-1]

	data_fx = itp.interp1d(np.arange(len(smooth_data))/60.0,smooth_data)
	

	print "done loading data"

	# Original frequency
	# Interpolation

	function_dict = {'exp': (np.exp, 0.5), 'log': (np.log,2.0), 'sig': (expit,10.0), '': (lambda x : x, 32.0)}

	if not os.path.exists(SOUND_DIR) : os.mkdir(SOUND_DIR)
	

	if (any (map(lambda x : x in function_dict, sys.argv))) :
		key = filter(lambda x : x in function_dict, sys.argv)[0]
	else:
		key = ''


	function, factor = function_dict[key]
	print((key, factor))


	wav_samplerate = 44100.0

	# opening wav file
	wavef = wave.open(os.path.join(SOUND_DIR,"interpolate_merged_"+key+".wav"),'w')
	wavef.setnchannels(1)
	wavef.setsampwidth(2) 
	wavef.setframerate(wav_samplerate)
	


	max_x = len(smooth_data)/60 if len(sys.argv) <=2  else int(sys.argv[2])
	print max_x

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

	while x_val<max_x:
		y_val  = data_fx(x_val)
		yvals.append(y_val)
		y_add  = function(y_val*factor) #rescaled and transformed
		y_add  = y_add/myRange*(MAX_FREQ-MIN_FREQ)+MIN_FREQ # scale from one range to another
		y_cum += y_add*x_val # f(t)*t
		ycums.append(y_cum)
		amp    = np.sin(y_cum*2.0*np.pi)
		x_val += step
		wavef.writeframesraw(struct.pack('<h',amp*32767))

	wavef.writeframes('')
	wavef.close()	
