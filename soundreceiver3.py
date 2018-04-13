import socket, pyaudio, threading, sys, pickle, pygame, time, struct
from scipy import interpolate as itp
from scipy.interpolate import CubicSpline
from port import *
from numpy import *
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



def load_sound(filename):
	"""
    Load the sound file into numpy uint16 array.

    Parameters
    ----------
    filename : wav file.

    Returns
    -------
    sif : numpy uint16 array
    """
	wav = wave.open(filename,'r')
	sig = wav.readframes('-1')
	sif = np.fromstring(sig,'int16')
	return sif



######## Global Variables ########


# timestamp
global timestamp
timestamp = None

# data buffer where frequency is received
global data
data = [0 for i in range(60)]

# global queue for sound buffer
global queue
queue = []


def add_to_data():	
	"""
    Function ran on separate thread for data fetching.
    Retrieves the data from 7011 port and add to data buffer for interpolation in the main thread.

    """
	subIndex = left_hand_col_indices(False)
	t0 = datetime.now()
	with ANReader(10.0,subIndex,port=7011,verbose=True,port_buffer_size=1024,recent_buffer_size=(10.0+1)*30) as reader:
		prevv = []
		while True:
			v,t,tAsDate = reader.copy_recent()
			if len(v)>0 and not (len(v)==len(prevv) and np.sum((v-prevv).flatten()**2)==0): # do we have enough data points?
			    avatar_velocity = fetch_matching_avatar_vel(avatar,np.array(tAsDate),t0)
			    diff = np.linalg.norm(avatar_velocity - v) # Send this data to streamer
			    data.append(diff*20)



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






if __name__ == '__main__':	

	# duration of sound streaming for each window
	# if duration = d, each window plays d seconds of sound
	duration = 1 if len(sys.argv) <2 else int(sys.argv[1])	

	# initial phase offset
	phaseOffset = 0


	# Open up the stream
	p = pyaudio.PyAudio()


	def callback(in_data, frame_count, time_info, status):
		"""
	    callback function for streamer
	    Reads from the data buffer if data buffer is not empty, else produce empty sound.

	    """
		dd = queue.pop() if len(queue)>0 else ''.join(map(lambda x : struct.pack('<h',x), np.array([0 for i in range(44100*duration)])))
		return (dd, pyaudio.paContinue)

	
	
	stream = p.open(format=8,
	                channels=1,
	                rate=44100,
	                frames_per_buffer=44100*duration, # How many frames are you going to play per buffer?
	                output=True,
	                stream_callback=callback)


	# Start collecting data
	addThread = threading.Thread(target=add_to_data)
	addThread.start()

	playstamp = time.time()
	stream.start_stream()

	snd = []


	datalength = 60

	# how many seconds per each point?
	sec_per_point = duration*1.0/datalength

	# initial tail should be 0
	# tail is the last value of previous window that would be interpolated with the first value of current window
	tail = 0

	yprev = []
	ycurr = []

	while True:
		if data is not None and len(data) >= datalength*2 :
			print "creating sound at " + str(time.time()-playstamp)
			itpval = data[:datalength*2]
			x = np.arange(datalength*2)
			xFinelySpaced=linspace(x[0],x[-1],44100)
			print "xFinelySpaced : " + str(len(xFinelySpaced))
			# print data
			y1 = itpval[:datalength]
			# print y1
			y2 = itpval[datalength:]
			# print y2

			cubicParams = spline(x[:datalength],y1)
			y1FinelySpaced=polyval(cubicParams,xFinelySpaced)

			ix=(xFinelySpaced>=datalength-1)&(xFinelySpaced<=datalength)

			xSpan=concatenate((xFinelySpaced[ix],x[datalength:]))
			ySpan=concatenate((y1FinelySpaced[ix],y2))
			dySpan=concatenate((polyval(polyder(cubicParams),xFinelySpaced[ix]),zeros(datalength)+nan))

			weightsSpan=concatenate(( ones(ix.sum())*2,ones(datalength) ))*datalength
			dweightsSpan=concatenate(( ones(ix.sum())*2,ones(datalength) ))

			splineParams=spline(xSpan,ySpan,4,
			                    fit_derivative=True,dy=dySpan,weights=weightsSpan)

			y2FinelySpaced=polyval(splineParams,xFinelySpaced)

			# print y2FinelySpaced

			ix=xFinelySpaced<datalength
			yCombined=concatenate((y1FinelySpaced[ix],y2FinelySpaced[ix==0]))[:44100]

			
			# plt.title("window " + str(index) + ", window " + str(index+1))

			# if index%2==0:
			# 	plt.plot(x[:datalength],y1,'g-')
			# 	plt.plot(x[datalength:],y2,'b-')
			# 	plt.plot(xFinelySpaced[:44100],yCombined[:44100],'k-')
			# 	plt.plot(xFinelySpaced[44100:],yCombined[44100:],'r-')
			# 	plt.show()
			# else :
			# 	plt.plot(x[:datalength],y1,'b-')
			# 	plt.plot(x[datalength:],y2,'g-')
			# 	plt.plot(xFinelySpaced[:44100],yCombined[:44100],'r-')
			# 	plt.plot(xFinelySpaced[44100:],yCombined[44100:],'k-')
			# 	plt.show()

			ycum = np.cumsum(yCombined)*2.0*np.pi/44100

			
			# debugging
			# if snd!=[]:
			# 	yprev = snd[-500:]


			snd = (32767.0*np.sin(ycum))

			# plt.plot(xFinelySpaced[44000:45000],snd[44000:45000],'k-')
			# plt.show()
			# if index%2==0:
			# 	plt.plot(xFinelySpaced[44000:44100],snd[44000:44100],'k-')
			# 	plt.plot(xFinelySpaced[44100:45000],snd[44100:45000],'r-')
			# 	plt.show()
			# else :
			# 	plt.plot(xFinelySpaced[44000:44100],snd[44000:44100],'r-')
			# 	plt.plot(xFinelySpaced[44100:45000],snd[44100:45000],'k-')
			# 	plt.show()

			val = ''.join(map(lambda x : struct.pack('<h',x), snd))
			
			# Length 2 string per each data point when written into byte stream
			queue+=(list(chunks(val,44100*duration*2)))

			del data[:datalength]
			# index += 1

	stream.stop_stream()
	addThread._stop_event.set()


		

    