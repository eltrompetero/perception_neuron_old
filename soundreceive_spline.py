import numpy as np, matplotlib.pyplot as plt, sys, threading, struct, pyaudio, time
from numpy import *
from port import *
from scipy.optimize import minimize
from scipy import interpolate as itp
from scipy.interpolate import CubicSpline
from datetime import datetime
from axis_neuron import left_hand_col_indices,right_hand_col_indices


def load_avatar(reverse_time=False,return_subject=False):
	"""
	This loads the correct avatar for comparison of performance. The handedness of the subject is read in
	from left_or_right.txt.

	Parameters
	----------
	reverse_time : bool,False
		If True, play avatar motion backwards in time.
	return_subject : bool,False

	Returns
	-------
	avatar : dict
		Dictionary of avatar interpolation splines.
	"""
	from pipeline import extract_motionbuilder_model3_3
	handedness = 'left'# handedness = open('%s/%s'%(DATADR,'left_or_right')).readline().rstrip()

	if handedness=='left':
		v,t = extract_motionbuilder_model3_3('Right',reverse_time=reverse_time)
	elif handedness=='right':
		v,t = extract_motionbuilder_model3_3('Left',reverse_time=reverse_time)
	else:
		print handedness
		raise Exception
	return v


verbose = ''
hand = 'left'

avatar = load_avatar(False, False)

def fetch_matching_avatar_vel(avatar,t,t0=None,verbose=False):
	"""
	Get the stretch of avatar velocities that aligns with the velocity data of the subject. 

	Parameters
	----------
	avatar : Interpolation
		This would be the templateTrial loaded in VRTrial.
	t : array of floats or datetime objects
		Stretch of time to return data from. If t0 is specified, this needs to be datetime objects.
	t0 : datetime,None
	verbose : bool,False

	Returns
	-------
	v : ndarray
		(n_time,3). Avatar's velocity that matches given time stamps relative to the starting time
		t0.
	"""
	if not t0 is None:
		# Transform dt to time in seconds.
		t = np.array([i.total_seconds() for i in t-t0])
		assert (t>=0).all()
	if verbose:
		print "Getting avatar times between %1.1fs and %1.1fs."%(t[0],t[-1])

	# Return part of avatar's trajectory that agrees with the stipulated time bounds.
	return avatar(t)




HOST = '127.0.0.1'   # use '' to expose to all networks




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
    Add random data to  databuffer for interpolation in the main thread.

    """
	start = time.time()
	while True:
		databuffer.append(np.random.choice(datas))


def add_to_data_avatar():	
	"""
	Function ran on separate thread for data fetching.
	Retrieves the data from 7011 port and add to databuffer for interpolation in the main thread.

	"""
	subIndex = left_hand_col_indices(False)
	t0 = datetime.now()
	with ANReader(10.0,subIndex,port=7011,verbose=True,port_buffer_size=1024,recent_buffer_size=(10.0+1)*30) as reader:
		prevv = []
		while True:
			v,t,tAsDate = reader.copy_recent()
			if len(v)>0 and not (len(v)==len(prevv) and np.sum((v-prevv).flatten()**2)==0): # do we have enough databuffer points?
				avatar_velocity = fetch_matching_avatar_vel(avatar,np.array(tAsDate),t0)
				diff = np.linalg.norm(avatar_velocity - v)
				databuffer.append(diff*20)


def print_queue():
	while True:
		if queue is not None and len(queue)>1:
			pop = queue.pop()


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
	



	##### GLOBAL VARIABLES #####


	# amount of data points per window
	data_per_window = 60

	# sampling frequency for velocity
	sample_freq = 60.0

	# how long does each sample take?
	sec_per_sample = data_per_window / sample_freq

	# unit duration
	duration = 1.0

	# sound sampling rate
	wav_rate = 44100




	# buffer that contains data
	global databuffer
	databuffer = []

	# buffer for storing int16 data points for sound
	global soundbuffer
	soundbuffer = []

	# buffer for storing string chunks of size frame_per_buffer converted from soundbuffer
	# Fed into callback function
	global queue
	queue = []


	# Initial calculation for getting the frames per buffer. 
	# This is a size of second window between two interpolated windows.
	# Stream should play the exact amount of frames that is fed into the queue at each time segment.
	x = np.arange(data_per_window*2)
	xFinelySpaced=linspace(x[0],x[-1],int(wav_rate*sec_per_sample))	
	ix=xFinelySpaced<data_per_window-0.5
	frames_per_buffer = len(xFinelySpaced[ix])
	frameOffset = len(xFinelySpaced[ix==0])



	# Start adding data to the global array
	# execute with first argument as value other than 1 for debugging
	addThread = threading.Thread(target=add_to_data_avatar) if len(sys.argv)>2 and sys.argv[1]=='1' else threading.Thread(target=add_to_data)
	addThread.start()



	start = time.time()


	def callback(in_data, frame_count, time_info, status):
		global lastdata
		"""
		callback function for streamer
		Reads from the databuffer buffer if databuffer buffer is not empty, else produce empty sound.
		"""

		# If queue is not empty, play the first chunk in the queue.
		# Queue contains the chunk of byte stream converted from interpolated sound databuffer.
		try:
			play_data = queue.pop() if len(queue)>0 else lastdata
			lastdata = play_data
		except NameError:
			lastdata = ''.join(map(lambda x : struct.pack('<h',x), np.array([0 for i in range(int(wav_rate*sec_per_sample))])))
			play_data = queue.pop() if len(queue)>0 else lastdata
			lastdata = play_data

		return (play_data, pyaudio.paContinue)






	p = pyaudio.PyAudio()	

	


	# Enable running script without sound streaming for debugging
	if len(sys.argv) > 2:
		stream = p.open(format=8,
						channels=1,
						rate=44100,
						frames_per_buffer=frames_per_buffer,
						output=True,
						stream_callback=callback)


		playstamp = time.time()
		stream.start_stream()		


	# Phase offset between two signals
	phaseOffset = 0

	
	# Previous parameter
	cprev = []

	# Previous play array (first window)
	pprev = []

	# Previous interpolated value
	yconprev = []


	while True:
		'''
		For interpolation, bring two windows of databuffer w_{t} and w_{t+1} at time frame t
		At every time step, given we already have cubic parameter for databuffer in w_{t},
		calculate parameter for w_{t+1} and concatenate to databuffer
		'''
		if databuffer is not None and len(databuffer) >= data_per_window*2 :

			x = np.arange(data_per_window*2)
			xFinelySpaced=linspace(x[0],x[-1],int(wav_rate*sec_per_sample))	

			# bring both windows from databuffer
			itpval = databuffer[:data_per_window*2]

			# w_{t}
			y1 = itpval[:data_per_window]

			# w_{t+1}
			y2 = itpval[data_per_window:]


			# If there is no previous parameter provided, create a new one
			if cprev==[]:
				cubicParams = spline(x[:data_per_window],y1,4)
				y1FinelySpaced=polyval(cubicParams,xFinelySpaced)
			else:
				# use the parameter for w_{t} if given
				cubicParams = cprev
				y1FinelySpaced= concatenate((polyval(cubicParams,xFinelySpaced)[-frameOffset:],
					np.zeros(int(wav_rate*sec_per_sample)-frameOffset)+polyval(cubicParams,xFinelySpaced)[-1]))

			

			# midpoints for preserving derivatives
			ix=(xFinelySpaced>=data_per_window-1)&(xFinelySpaced<=data_per_window)

			xSpan=concatenate((xFinelySpaced[ix],x[data_per_window:]))
			ySpan=concatenate((y1FinelySpaced[ix],y2))

			# calculating derivatives until the first window
			dySpan=concatenate((polyval(polyder(cubicParams),xFinelySpaced[ix]),zeros(data_per_window)+nan))

			weightsSpan=concatenate(( ones(ix.sum())*2,ones(data_per_window) ))*data_per_window
			dweightsSpan=concatenate(( ones(ix.sum())*2,ones(data_per_window) ))


			splineParams=spline(xSpan,ySpan,4,fit_derivative=True,dy=dySpan,weights=dweightsSpan)


			# Calculate y2 given spline params
			y2FinelySpaced=polyval(splineParams,xFinelySpaced)

			# Combine interpolation
			ix1 = (xFinelySpaced < data_per_window-0.5)
			yCombined=concatenate((y1FinelySpaced[ix1],y2FinelySpaced[ix1==0]))

			# We should only be playing the first window
			play_array = y1FinelySpaced[ix1]
			ycum = np.cumsum(play_array)*2.0*np.pi/(int(wav_rate))
			soundbuffer = (32767.0*np.sin(ycum+phaseOffset))
			phaseOffset = ycum[-1]
			val = ''.join(map(lambda x : struct.pack('<h',x), soundbuffer))
			
			# Length 2 string per each databuffer point when written into byte stream
			queue+=(list(chunks(val,int(wav_rate*sec_per_sample*2))))

			if len(databuffer) >= data_per_window * 2:
				del databuffer[:data_per_window]

			# store parameter for w_{t+1}
			cprev = splineParams			
			yconprev = yCombined			

			print "it took " + str(time.time()-start) + " seconds"
			start = time.time()



