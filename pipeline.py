# Module for quick access to analysis pipelines.
# 2017-03-31
from __future__ import division
import pickle
import numpy as np
from load import *
from utils import *
from filter import *

def quick_load(fileix,dt=1/120,negate_x=True,negate_y=False,disp=True):
    """
    Quick and dirty method for loading filtered velocities from hand trials. Currently returning the
    y-velocities only.

    Returns:
    --------
    T
    v1
        y-axis
    v2
        y-axis
    """
    bodypartix = hand_ix(fileix)
    fname = get_fnames()[fileix]
    if type(fname) is tuple:
        fname,date = fname
    else:
        date = None

    data = pickle.load(open('%s/%s.p'%(get_dr(fname,date),fname),'rb'))
    leaderX,leaderV,leaderA = data['leaderX'],data['leaderV'],data['leaderA']
    followerX,followerV,followerA = data['followerX'],data['followerV'],data['followerA']
    
    T = np.arange(len(leaderX[0]))*dt

    x1,x2 = leaderX[bodypartix],followerX[bodypartix]
    v1,v2 = leaderV[bodypartix],followerV[bodypartix]
    a1,a2 = leaderA[bodypartix],followerA[bodypartix]
    
    if negate_x:
        v2[:,0] *= -1
    if negate_y:
        v2[:,1] *= -1
    if disp:
        print np.corrcoef(v1[:,0],v2[:,0])[0,1]
        print np.corrcoef(v1[:,1],v2[:,1])[0,1]

    # Detrending necessary in some cases.
    if fileix in [51,52,53]:
        detrend(v2[:,2],inplace=True)
    return T,v1,v2

def pipeline_phase_calc(fileixs=[],
                        trajs=[],
                        suffix='',
                        down_sample=False,
                        **phase_calc_kwargs):
    """
    Pipeline loading pickled lowpass filtered data and running phase extraction after bandpass filtering.
    Pickles list of tuples phases, list of tuples vs (containing filtered velocities), and array fs
    (frequencies that were bandpassed).  Pickles are saved in phase_files/phase_%d.p

    If fileix's are given, those are loaded then processed. Otherwise, the data given is processed.
    
    Params:
    -------
    fileix (list)
        List of ints of files to load.
    trajs (list of tuples)
        (T,v1,v2)
    sample_freq (str='120')
        '120' or '60'
    bandwidth (float=.1)
        Bandwidth of bandpass filter.
    down_sample (bool=False)
        Down sample data by a factor of 2 if true.
    """
    fs = np.concatenate((np.arange(-3,0,.1),np.arange(.1,3.1,.1)))
    
    if len(fileixs)>0:
        for fileix in fileixs:
            print "Starting file %d..."%fileix
            T,v1,v2 = quick_load(fileix,dt=1/int(sample_freq))
            if down_sample:
                T = T[::2]
                v1 = v1[::2]
                v2 = v2[::2]

            phases,vs = phase_calc(fs,v1,v2,**phase_calc_kwargs) 

            pickle.dump({'phases':phases,'vs':vs,'fs':fs},open('phase_files/phase_%d%s.p'%(fileix,suffix),'wb'),-1)
            print "Done with file %d."%fileix
    else:
        counter = 0
        for T,v1,v2 in trajs:
            assert T.ndim==1 and v1.ndim==1 and v2.ndim==1
            if down_sample:
                T = T[::2]
                v1 = v1[::2]
                v2 = v2[::2]

            phases,vs = phase_calc(fs,v1,v2,**phase_calc_kwargs) 

            pickle.dump({'phases':phases,'vs':vs,'fs':fs},
                        open('phase_files/temp_phase_%d%s.p'%(counter,suffix),'wb'),-1)
            print "Done with file %d."%fileix
            counter += 1

def phase_calc(fs,v1,v2,
               sample_freq='120',
               bandwidth=.1,
               down_sample=False):
    """
    Params:
    -------
    fs (ndarray)
        Frequencies at which to bandpass.
    v1,v2 (ndarray)
        1d arrays.
    sample_freq (str='120')
        '120' or '60'
    bandwidth (float=.1)
        Bandwidth of bandpass filter.
    down_sample (bool=False)
        Down sample data by a factor of 2 if true.
    """
    from scipy.signal import hilbert

    if str(sample_freq)=='120':
        windowLength,filtwidth = 501,50
    else:
        windowLength,filtwidth = 251,25
    
    phases = []
    vs = []
    for f in fs:
        v1_ = moving_freq_filt(v1,window=windowLength,
                               window_type=('gaussian',filtwidth),
                               filter_type='single',
                               sample_freq=int(sample_freq),
                               pass_freq=f,
                               bandwidth=bandwidth,
                               axis=0)
        v2_ = moving_freq_filt(v2,window=windowLength,
                               window_type=('gaussian',filtwidth),
                               filter_type='single',
                               sample_freq=int(sample_freq),
                               pass_freq=f,
                               bandwidth=bandwidth,
                               axis=0)
        
        h1 = hilbert(v1_,axis=0)
        h2 = hilbert(v2_,axis=0)
        phase1 = np.angle(h1)
        phase2 = np.angle(h2)

        phases.append((phase1,phase2))
        vs.append((v1_,v2_))
    
    return phases,vs
