import numpy as np, matplotlib.pyplot as plt, threading, struct, pyaudio, time
from numpy import *
from scipy.optimize import minimize
from scipy import interpolate as itp
from scipy.interpolate import CubicSpline


##### GLOBAL VARIABLES #####


# amount of data per window
data_per_window = 60

# sampling frequency for velocity
sample_freq = 60.0

# how long does each sample take?
sec_per_sample = data_per_window / sample_freq


# unit duration
duration = 1.0

# sound sampling rate
wav_rate = 44100


global data
data = [0 for i in range(1)]

global lastdata
lastdata = ''.join(map(lambda x : struct.pack('<h',x), np.array([0 for i in range(int(wav_rate*sec_per_sample))])))

# global queue for sound buffer
global queue
queue = []

datas = [440, 880, 220, 550, 330]


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]




def nan_norm(y):
    """Calculate L2 norm of given vector ignoring nans."""
    return sqrt( nansum(y**2) )



def spline(x,y,order=3,fit_derivative=False,dy=None,weights=None,dy_weights=None):
    """Fit spline. You may need to optimize this by rewriting the minimization routine.
    
    Parameters
    ----------
    x : ndarray
        x-values to fit
    y : ndarray
        y-values for each x value
    order : int,3
        Order of spline.
    fit_derivative : bool,False
        If True, fit the first derivative as well. This must be specified in dy.
    dy : ndarray,None
        First derivative for each point given in y. There can be nan values where 
        there is no derivative defined.
    weights : ndarray,None
        Weighting function for each y.
    dy_weights : ndarray,None
        Optional separate weighting function for the derivatives.
        
    Returns
    -------
    spline_parameters : ndarray
        The coefficients for the spline ordered from highest degree to lowest
        a_n * x^n + a_{n-1} * x^{n-1} + ... + a0. This can be given to numpy's 
        polyval routine.
    """
    if weights is None:
        weights=ones_like(x)
    if dy_weights is None:
        dy_weights=weights
        
    if not fit_derivative:
        return minimize(lambda params:np.linalg.norm( (polyval(params,x)-y)*weights ),
                        zeros(order+1))['x']
    else:
        return minimize(lambda params:np.linalg.norm( (polyval(params,x)-y)*weights )+
                        nan_norm( (polyval(polyder(params),x)-dy)*dy_weights ),
                        zeros(order+1))['x']






def add_to_data():	
	"""
    Function ran on separate thread for data fetching.
    Retrieves the data from 7011 port and add to data buffer for interpolation in the main thread.

    """
	start = time.time()
	while True:
		# print "add"
		data.append(np.random.choice(datas))
			# print "it took " + str(time.time() - start) + " seconds to add 120 data"
			# start = time.time()
		# print data


def print_queue():
	
	while True:
		if queue is not None and len(queue)>1:
			pop = queue.pop()
			print "queue"


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, target=None):
        super(StoppableThread, self).__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

if __name__ == '__main__':
	

	addThread = threading.Thread(target=add_to_data)
	addThread.start()

	# sound buffer
	global snd
	snd = []

	# for debugging
	yprev = []

	index = 0

	start = time.time()


	def callback(in_data, frame_count, time_info, status):
		"""
		callback function for streamer
		Reads from the data buffer if data buffer is not empty, else produce empty sound.
		"""

		# If queue is not empty, play the first chunk in the queue.
		# Queue contains the size 44100 chunk of byte stream converted from interpolated sound data.
		# print "queue : " + str(len(queue))
		try:
			play_data = queue.pop() if len(queue)>0 else lastdata
			lastdata = play_data
		except NameError:
			print " no lastdata "
			global lastdata
			lastdata = ''.join(map(lambda x : struct.pack('<h',x), np.array([0 for i in range(int(wav_rate*sec_per_sample))])))
			play_data = queue.pop() if len(queue)>0 else lastdata
			lastdata = play_data
		return (play_data, pyaudio.paContinue)


	p = pyaudio.PyAudio()	
	stream = p.open(format=8,
					channels=1,
					rate=44100,
					frames_per_buffer=(int(wav_rate*sec_per_sample)), # How many frames are you going to play per buffer?
					output=True,
					stream_callback=callback)


	playstamp = time.time()
	stream.start_stream()		

	while True:
		if data is not None and len(data) >= data_per_window :
			itpval = data[:data_per_window*2]
			x = np.arange(data_per_window*2)
			xFinelySpaced=linspace(x[0],x[-1],int(wav_rate*sec_per_sample))
			print "xFinelySpaced : " + str(len(xFinelySpaced))
			# print data
			y1 = itpval[:data_per_window]
			# print y1
			y2 = itpval[data_per_window:]
			# print y2

			cubicParams = spline(x[:data_per_window],y1)
			y1FinelySpaced=polyval(cubicParams,xFinelySpaced)

			ix=(xFinelySpaced>=data_per_window-1)&(xFinelySpaced<=data_per_window)

			xSpan=concatenate((xFinelySpaced[ix],x[data_per_window:]))
			ySpan=concatenate((y1FinelySpaced[ix],y2))
			dySpan=concatenate((polyval(polyder(cubicParams),xFinelySpaced[ix]),zeros(data_per_window)+nan))

			weightsSpan=concatenate(( ones(ix.sum())*2,ones(data_per_window) ))*data_per_window
			dweightsSpan=concatenate(( ones(ix.sum())*2,ones(data_per_window) ))

			splineParams=spline(xSpan,ySpan,4,
			                    fit_derivative=True,dy=dySpan,weights=weightsSpan)

			y2FinelySpaced=polyval(splineParams,xFinelySpaced)

			# print y2FinelySpaced

			ix=xFinelySpaced<data_per_window
			yCombined=concatenate((y1FinelySpaced[ix],y2FinelySpaced[ix==0]))

			
			# plt.title("window " + str(index) + ", window " + str(index+1))

			# if index%2==0:
			# 	plt.plot(x[:data_per_window],y1,'g-')
			# 	plt.plot(x[data_per_window:],y2,'b-')
			# 	plt.plot(xFinelySpaced[:int(wav_rate*sec_per_sample)],yCombined[:int(wav_rate*sec_per_sample)],'k-')
			# 	plt.plot(xFinelySpaced[int(wav_rate*sec_per_sample):],yCombined[int(wav_rate*sec_per_sample):],'r-')
			# 	plt.show()
			# else :
			# 	plt.plot(x[:data_per_window],y1,'b-')
			# 	plt.plot(x[data_per_window:],y2,'g-')
			# 	plt.plot(xFinelySpaced[:int(wav_rate*sec_per_sample)],yCombined[:int(wav_rate*sec_per_sample)],'r-')
			# 	plt.plot(xFinelySpaced[int(wav_rate*sec_per_sample):],yCombined[int(wav_rate*sec_per_sample):],'k-')
			# 	plt.show()

			ycum = np.cumsum(yCombined)*2.0*np.pi/(int(wav_rate*sec_per_sample))

			
			# debugging
			# if snd!=[]:
			# 	yprev = snd[-500:]


			snd = (32767.0*np.sin(ycum))

			# plt.plot(xFinelySpaced[700:800],snd[700:800],'b-')
			# if index%2==0:
			# 	plt.plot(xFinelySpaced[700:735],snd[700:735],'k-')
			# 	plt.plot(xFinelySpaced[735:800],snd[735:800],'r-')
			# 	plt.show()
			# else :
			# 	plt.plot(xFinelySpaced[700:735],snd[700:735],'r-')
			# 	plt.plot(xFinelySpaced[735:800],snd[735:800],'k-')
			# 	plt.show()

			val = ''.join(map(lambda x : struct.pack('<h',x), snd))
			
			# Length 2 string per each data point when written into byte stream
			queue+=(list(chunks(val,int(wav_rate*sec_per_sample*2))))

			if len(data) >= data_per_window * 2:
				del data[:data_per_window]

			index += 1

			print "it took " + str(time.time()-start) + " seconds"
			start = time.time()

	addThread._stop_event.set()



