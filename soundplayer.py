import os, wave, struct, math, sys, numpy as np, matplotlib.pyplot as plt
from pipeline import extract_motionbuilder_model3_3
from itertools import izip
from scipy import interpolate as itp
from scipy.special import expit
import numpy.fft as fft


SOUND_DIR = "soundfiles"


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

	wav_samplerate = 44100.0

	f = lambda x : x*240

	data_chunks = list(chunks(data_amp,600))

	# Original frequency
	# Interpolation

	function_dict = {'exp': np.exp, 'log': np.log, 'sig': expit}

	
	wavef = wave.open(os.path.join(SOUND_DIR,"interpolate_merged.wav",'w'))
	wavef.setnchannels(1)
	wavef.setsampwidth(2) 
	wavef.setframerate(wav_samplerate)
	
	data_fx = itp.interp1d(range(len(data_amp)),data_amp)

	cumm = 0
	max_x = len(data_amp) if len(sys.argv) <= 1 else int(sys.argv[1])

	step = 60.0/44100.0
	x_val = 0.0
	y_cum = 0


	while x_val<max_x :
		y_val  = data_fx(x_val)
		y_cum += (y_val*2.0*np.pi)/32.0
		amp    = np.sin(y_cum)
		x_val += step
		wavef.writeframesraw(struct.pack('<h',amp*32767))

	wavef.writeframes('')
	wavef.close()	


	# First dividing and then interpolating

	# wavef = wave.open("../interpolate_exp.wav",'w')
	# wavef.setnchannels(1)
	# wavef.setsampwidth(2) 
	# wavef.setframerate(wav_samplerate)

	# for i, data_chunk in list(enumerate(data_chunks))[:1]:
	# 	newx = np.linspace(0,len(data_chunk)-1,wav_samplerate*len(data_chunk)/60)
	# 	newy = data_fx(newx)
	# 	freq = newy
	# 	freq_time = [a*b for a,b in zip(freq,np.linspace(1,len(data_chunk)/60,wav_samplerate*len(data_chunk)/60+1))]
	# 	freq_2pi = map(lambda x : 2.0*np.pi*np.exp(x)/500, freq_time)
	# 	# save_sound("../interpolate.wav",amp,44100,wav_samplerate=44100)
	# 	freq_csed   = np.cumsum(np.array(freq_2pi))
	# 	amp3 = np.sin(freq_csed)
	# 	cumm = freq_csed[-1]
	# 	for value in amp3:
	# 		wavef.writeframesraw(struct.pack('<h',value*32767))	

	# wavef.writeframes('')
	# wavef.close()
