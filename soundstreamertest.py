
from port import *
from datetime import datetime
from axis_neuron import left_hand_col_indices,right_hand_col_indices
import soundstreamer

reload(soundstreamer)

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


streamer = soundstreamer.SoundStreamer()  
streamer.init()



def read_and_compare():
    subIndex = left_hand_col_indices(False)
    t0 = datetime.now()

    def add_data():
        with ANReader(10.0,subIndex,port=7011,verbose=True,port_buffer_size=1024,recent_buffer_size=(10.0+1)*30) as reader:
            prevv = []
            while True:
                v,t,tAsDate = reader.copy_recent()
                if len(v)>0 and not (len(v)==len(prevv) and np.sum((v-prevv).flatten()**2)==0): # do we have enough data points?
                    avv = fetch_matching_avatar_vel(avatar,np.array(tAsDate),t0)
                    diff = np.linalg.norm(avv - v) # Send this data to streamer
                    streamer.addToStream(diff*100)
                prevv = v
    
    addThread = threading.Thread(target=add_data)
    addThread.start()
    ipdThread = threading.Thread(target = streamer.interpolate)
    ipdThread.start()
    playThread = threading.Thread(target = streamer.playSound)
    playThread.start()
    
    
        
            # else :
            #     print "len v == 0"
            # streamer.update()


        	    # streamer.




