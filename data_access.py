# ================================================================================================ # 
# Module for keeping track of processed data that is ready for analysis. Some classes for organizing
# data are included.
# 
# Author: Eddie Lee edl56@cornell.edu
# ================================================================================================ # 


from .utils import *
import os
import pickle as pickle
from warnings import warn



def subject_settings_v3(index,return_list=True):
    settings = [{'person':'Zimu3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Darshna3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Richard3',
                  'modelhandedness':['Right','Left','Right','Left'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Rachel3',
                  'modelhandedness':['Right','Left','Right','Left'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Priyanka3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Emily3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Sam3',
                  'modelhandedness':['Right','Left','Right','Left'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Najila3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Kemper3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Lauren3',
                  'modelhandedness':['Right','Left','Right','Left'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']}
                ][index]
    dr = '../data/UE4_Experiments/%s'%settings['person']
    if return_list:
        output = [settings[k] for k in ['person','modelhandedness','rotation']]
        output.append(dr)
        return output
    return settings,dr

def subject_settings_v3_1(index,return_list=True):
    """
    Subject info for experiment v3.1.

    Parameters
    ----------
    index : int
    return_list : bool,True

    Returns
    -------
    settings : dict
    dr : str
    """
    settings = [{'person':'Subject01_3_1',
                  'modelhandedness':['Left'],
                  'rotation':[0],
                  'trials':['avatar']},
                {'person':'Subject02_3_1',
                  'modelhandedness':['Left'],
                  'rotation':[0],
                  'trials':['avatar']},
                {'person':'Subject03_3_1',
                  'modelhandedness':['Left'],
                  'rotation':[0],
                  'trials':['avatar']},
                {'person':'Subject04_3_1',
                  'modelhandedness':['Right'],
                  'rotation':[0],
                  'trials':['avatar']},
                {'person':'Subject05_3_1',
                  'modelhandedness':['Right'],
                  'rotation':[0],
                  'trials':['avatar']}
                ][index]
    dr = '../data/UE4_Experiments/%s'%settings['person']
    if return_list:
        output = [settings[k] for k in ['person','modelhandedness','rotation']]
        output.append(dr)
        return output
    return settings,dr

def subject_settings_v3_2(index,return_list=True):
    """
    Subject info for experiment v3.2.

    Parameters
    ----------
    index : int
    return_list : bool,True

    Returns
    -------
    settings : dict
    dr : str
    """
    settings = [{'person':'Subject01_3_2',
                 'rotation':[0],
                 'trials':['avatar']},
                {'person':'Subject02_3_2',
                 'rotation':[0],
                 'trials':['avatar']}
                ][index]
    dr = '../data/UE4_Experiments/%s'%settings['person']
    if return_list:
        output = [settings[k] for k in ['person','rotation']]
        output.append(dr)
        return output
    return settings,dr

def subject_settings_v3_3(index,hand,return_list=True):
    """
    Subject info for experiment v3.3. Twoples refer to the left and right subject hands.
    2018-01-15

    Parameters
    ----------
    index : int
    hand : str
        Subject hand used.
    return_list : bool,True

    Returns
    -------
    settings : dict
    dr : str
    """
    settings = [{'person':'Subject01_3_3',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,False]},
                {'person':'Subject02_3_3',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,True]},
                {'person':'Subject03_3_3',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,False]},
                {'person':'Subject04_3_3',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,True]},
                {'person':'Subject05_3_3',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,True]},
                {'person':'Subject06_3_3',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject07_3_3',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject08_3_3',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject09_3_3',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject10_3_3',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[False,True]},
                {'person':'Subject11_3_3',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,False]},
                {'person':'Subject12_3_3',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                ][index]
    dr = '../data/UE4_Experiments/%s/%s'%(settings['person'],hand)
    try:
        rotAngle = pickle.load(open('%s/%s'%(dr,'gpr.p'),'rb'),encoding='latin1')['rotAngle']
    except IOError:
        rotAngle=np.nan
    reverse=settings['reverse'][0] if hand=='left' else settings['reverse'][1]
    usable=settings['usable'][0] if hand=='left' else settings['usable'][1]

    if return_list:
        return settings['person'],dr,rotAngle,reverse,usable
    return settings,dr

def subject_settings_v3_4(index,hand,return_list=True):
    """
    Subject info for experiment v3.4. Audio no training. Twoples refer to the left and right subject
    hands.
    2018-02-16

    Parameters
    ----------
    index : int
    hand : str
        Subject hand used.
    return_list : bool,True

    Returns
    -------
    settings : dict
    dr : str
    rotAngle : float
    reverse : bool
    usable : bool
    """
    settings = [{'person':'Subject01_3_4',
                 'trials':['avatar'],
                 'reverse':[False,False],
                 'usable':[False,True]},
                {'person':'Subject02_3_4',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,False]},
                {'person':'Subject03_3_4',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[False,True]},
                {'person':'Subject04_3_4',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,False]},
                {'person':'Subject05_3_4',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,False]},
                {'person':'Subject06_3_4',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[False,True]},
                {'person':'Subject07_3_4',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,False]},
                {'person':'Subject08_3_4',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,False]},
                {'person':'Subject09_3_4',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,False]},
                {'person':'Subject10_3_4',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,False]},
                {'person':'Subject11_3_4',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[False,True]},
                {'person':'Subject12_3_4',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,False]}
                ][index]
    dr = '../data/UE4_Experiments/%s/%s'%(settings['person'],hand)
    try:
        rotAngle = pickle.load(open('%s/%s'%(dr,'gpr.p'),'rb'),encoding='latin1')['rotAngle']
    except IOError:
        rotAngle=np.nan
    reverse=settings['reverse'][0] if hand=='left' else settings['reverse'][1]
    usable=settings['usable'][0] if hand=='left' else settings['usable'][1]

    if return_list:
        return settings['person'],dr,rotAngle,reverse,usable
    return settings,dr

def subject_settings_v3_5(index,hand,return_list=True):
    """
    Subject info for experiment v3.5. Twoples refer to the left and right subject hands.
    2018-04-15

    Parameters
    ----------
    index : int
    hand : str
        Subject hand used.
    return_list : bool,True

    Returns
    -------
    settings : dict
    dr : str
    """
    settings = [{'person':'Subject01_3_5',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,True]},
                {'person':'Subject02_3_5',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,True]},
                {'person':'Subject03_3_5',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,True]},
                {'person':'Subject04_3_5',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[False,True]},
                {'person':'Subject05_3_5',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject06_3_5',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,True]},
                {'person':'Subject07_3_5',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,True]},
                {'person':'Subject08_3_5',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject09_3_5',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[False,True]},
                {'person':'Subject10_3_5',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject11_3_5',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject12_3_5',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject13_3_5',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject14_3_5',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,True]}
                ][index]
    dr = '../data/UE4_Experiments/%s/%s'%(settings['person'],hand)
    try:
        rotAngle = pickle.load(open('%s/%s'%(dr,'gpr.p'),'rb'),encoding='latin1')['rotAngle']
        # In the case where the final save in HandSyncExperiment.run_vr did not complete, the
        # rotAngle will be a list.
        if type(rotAngle) is list:
            rotAngle=rotAngle[0] if hand=='left' else rotAngle[1]
    except KeyError:
        from .experiment import HandSyncExperiment

        # Need to reload data from file.
        f=[f for f in os.listdir('%s'%dr) if 'an_port_cal' in f]
        f.sort()
        rotAngle=HandSyncExperiment.read_cal('%s/%s'%(dr,f[-1]),.3)
        rotAngle=rotAngle[0] if hand=='left' else rotAngle[1]
    except IOError:
        rotAngle=np.nan
    reverse=settings['reverse'][0] if hand=='left' else settings['reverse'][1]
    usable=settings['usable'][0] if hand=='left' else settings['usable'][1]

    if return_list:
        return settings['person'],dr,rotAngle,reverse,usable
    return settings,dr

def subject_settings_v3_6(index,hand,return_list=True):
    """
    Subject info for experiment v3.3. Twoples refer to the left and right subject hands.
    2018-04-22

    Parameters
    ----------
    index : int
    hand : str
        Subject hand used.
    return_list : bool,True

    Returns
    -------
    settings : dict
    dr : str
    """
    assert hand in ['left','right']
    settings = [{'person':'Subject01_3_6',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,True]},
                {'person':'Subject02_3_6',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject03_3_6',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject04_3_6',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,True]},
                {'person':'Subject05_3_6',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,False]},
                {'person':'Subject06_3_6',
                 'trials':['avatar'],
                 'reverse':[False,True],
                 'usable':[True,True]},
                {'person':'Subject07_3_6',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]},
                {'person':'Subject08_3_6',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[False,False]},
                {'person':'Subject09_3_6',
                 'trials':['avatar'],
                 'reverse':[True,False],
                 'usable':[True,True]}
                ][index]
    dr = '../data/UE4_Experiments/%s/%s'%(settings['person'],hand)
    try:
        rotAngle = pickle.load(open('%s/%s'%(dr,'gpr.p'),'rb'),encoding='latin1')['rotAngle']
    except IOError:
        rotAngle=np.nan
    reverse=settings['reverse'][0] if hand=='left' else settings['reverse'][1]
    usable=settings['usable'][0] if hand=='left' else settings['usable'][1]

    if return_list:
        return settings['person'],dr,rotAngle,reverse,usable
    return settings,dr




# ------------------ #
# Class definitions. #
# ------------------ #
class VRTrial3_1(object):
    def __init__(self,person,modelhandedness,rotation,dr,
                 fname='trial_dictionaries.p',
                 reverse=False,
                 retrain=True):
        """
        Parameters
        ----------
        person : str
        modelhandedness : list of str
        rotation : list of float
        dr : str

        Members
        -------
        person : str
        modelhandedness : list of str
        rotation : float
        dr : str
        subjectTrial : dict
            Full Axis Neuron trial data labeled by part+'T' part+'V'.
        templateTrial : dict
            Full MotionBuilder trial data labeled by part+'T' part+'V'.
        timeSplitTrials
        subjectSplitTrials
        templateSplitTrials

        Methods
        -------
        info
        subject_by_window_dur
        subject_by_window_spec
        pickle_trial_dicts
        pickle_phase
        _fetch_windowspec_indices
        """
        self.person = person
        self.modelhandedness = modelhandedness
        self.rotation = rotation
        self.dr = dr
        self.reverse = reverse

        # Load gpr data points.
        savedData = pickle.load(open('%s/%s'%(self.dr,'gpr.p'),'rb'),encoding='latin1')
        self.gprmodel = savedData['gprmodel']
        self.pause=savedData['pause'],savedData['unpause']
        self.trialTypes = ['avatar']
        
        try:
            data = pickle.load(open('%s/%s'%(self.dr,fname),'rb'))
        except Exception:
            self.pickle_trial_dicts(1)
            
        data = pickle.load(open('%s/%s'%(self.dr,fname),'rb'))
        self.templateTrial = data['templateTrial']
        self.subjectTrial = data['subjectTrial']
        self.timeSplitTrials = data['timeSplitTrials']
        self.templateSplitTrials = data['templateSplitTrials']
        self.subjectSplitTrials = data['subjectSplitTrials']
        self.windowsByPart = data['windowsByPart']
        
        if retrain:
            self.retrain_gprmodel()
    
    def info(self):
        print("Person %s"%self.person)
        print("Trials available:")
        for part in self.trialTypes:
            print("%s\tInvisible\tTotal"%part)
            for spec,_ in self.windowsByPart[part]:
                print("\t%1.2f\t\t%1.2f"%(spec[0],spec[1]))
    
    def subject_by_window_dur(self,windowDur,part):
        """
        Params:
        -------
        windowDur (list)
            Duration of visible/invisible cycle.
        part (str)
            Body part to return.
            
        Returns:
        --------
        selection (list)
            List of trials that have given window duration. Each tuple in list is a tuple of the 
            ( (invisible,total window), time, extracted velocity data ).
        """
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.subjectSplitTrials[part][i] ))
        return selection
    
    def template_by_window_dur(self,windowDur,part):
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.templateSplitTrials[part][i] ))
        return selection
    
    def subject_by_invisible_dur(self,windowDur,part):
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[0]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.subjectSplitTrials[part][i] ))
        return selection

    def subject_by_window_spec(self,windowSpec,trial_type,precision=None):
        """Automatically search through left and right hand trials."""
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            for ix_ in ix:
                selection.append(( self.windowsByPart[trial_type][ix_][0],
                                   self.timeSplitTrials[trial_type][ix_],
                                   self.subjectSplitTrials[trial_type][ix_].copy() ))
            # Iterate also through hand0 or avatar0, which contains the other hand..
            #if trial_type.isalpha():
            #    selection += self.subject_by_window_spec([windowSpec[specix]],
            #                                                 trial_type+'0',
            #                                                 precision=precision)
        return selection

    def template_by_window_spec(self,windowSpec,trial_type,precision=None):
        """Automatically search through left and right hand trials."""
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            for ix_ in ix:
                selection.append(( self.windowsByPart[trial_type][ix_][0],
                                   self.timeSplitTrials[trial_type][ix_],
                                   self.templateSplitTrials[trial_type][ix_].copy() ))
            # Iterate also through hand0 or avatar0, which contains the other hand..
            #if trial_type.isalpha():
            #    selection += self.template_by_window_spec([windowSpec[specix]],
            #                                                 trial_type+'0',
            #                                                 precision=precision)
        return selection

    def template_by_invisible_dur(self,windowDur,part):
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[0]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.templateSplitTrials[part][i] ))
        if trialType.isalpha():
            return selection + self.template_by_invisible_dur(windowSpec,trialType+'0')
        return selection

    def visibility_by_window_spec(self,windowSpec,trial_type,precision=None):
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.templateSplitTrials[trial_type+'visibility'][ix[0]] ))
            # Iterate also through hand0 or avatar0, which contains the other hand..
            #if trial_type.isalpha():
            #    selection += self.visibility_by_window_spec([windowSpec[specix]],
            #                                                 trial_type+'0',
            #                                                 precision=precision)
        return selection

    def phase_by_window_dur(self,source,windowDur,trialType):
        """
        Return instantaneous phase from bandpass filtered velocities on trial specificied by window
        duration.

        Params:
        -------
        source (str)
        windowDur (list of floats)
        trialType (str)
            'avatar', 'avatar0', 'hand', 'hand0'
        """
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i = 0
        for spec,_ in self.windowsByPart[trialType]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            try:
                if source=='subject' or source=='s':
                    phases = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trialType,i),
                                         'rb'))['phases']
                elif source=='template' or source=='t':
                    phases = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trialType,i),
                                         'rb'))['phases']

                phases = [np.vstack(p) for p in phases]
                selection.append(( self.windowsByPart[trialType][i][0],phases ))
            except IOError:
                print("Trial %d in trial type %s not found."%(i,trialType))
        return selection

    def phase_by_window_spec(self,source,windowSpec,trial_type):
        """
        Parameters
        ----------
        source : str
        windowSpec : list
        trial_type : str
        """
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.templateSplitTrials[trial_type][ix[0]] ))
                try:
                    if source=='subject' or source=='s':
                        data = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trial_type,ix[0]),'rb'))
                        phases,vs = data['phases'],data['vs']
                    elif source=='template' or source=='t':
                        data = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trial_type,ix[0]),'rb'))
                        phases,vs = data['phases'],data['vs']
                    
                    phases = [np.vstack(p) for p in phases]
                    selection.append(( self.windowsByPart[trial_type][ix[0]][0],phases ))
                except IOError:
                    print("Trial %d in trial type %s not found."%(ix[0],trial_type))

            # Iterate also through hand0 or avatar0, which contains the other hand.
            if trial_type.isalpha():
                selection += self.phase_by_window_spec(source,
                                                        [windowSpec[specix]],
                                                        trial_type+'0',
                                                        precision=precision)
        return selection

    def filtv_by_window_spec(self,source,windowSpec,trialType,search_all=True):
        """
        Returns:
        --------
        list of twoples (windowSpec, filtv) where filtv is a list of 3 arrays corresponding to each dimension
        """
        raise NotImplementedError()
        ix = self._fetch_windowspec_indices(windowSpec,trialType,precision=precision)
        selection = []

        for i in ix:
            if source=='subject' or source=='s':
                data = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))
                phases,vs = data['phases'],data['vs']
            elif source=='template' or source=='t':
                data = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))
                phases,vs = data['phases'],data['vs']
            else:
                raise Exception

            vs = [np.vstack(p) for p in vs]
            selection.append(( self.windowsByPart[trialType][i][0],vs ))

        if trialType.isalpha() and search_all:
            return selection + self.filtv_by_window_spec(source,windowSpec,trialType+'0',False)

        return selection

    def dphase_by_window_dur(self,windowDur,trialType):
        """
        Difference in phase between subject and template motion.
        """
        raise NotImplementedError
        from misc.angle import mod_angle
        
        subjectPhase = self.phase_by_window_dur('s',windowDur,trialType)
        templatePhase = self.phase_by_window_dur('t',windowDur,trialType)
        dphase = []
        
        for i in range(len(subjectPhase)):
            dphase.append(( subjectPhase[i][0], 
                            [mod_angle( s-t ) for s,t in zip(subjectPhase[i][1],templatePhase[i][1])] ))
        if trialType.isalpha():
            return dphase + self.dphase_by_window_dur(windowDur,trialType+'0')
        return dphase

    def dphase_by_window_spec(self,windowSpec,trialType):
        """
        Difference in phase between subject and template motion.
        """
        from misc.angle import mod_angle
        
        subjectPhase = self.phase_by_window_spec('s',windowSpec,trialType)
        templatePhase = self.phase_by_window_spec('t',windowSpec,trialType)
        dphase = []
            
        for i in range(len(subjectPhase)):
            dphase.append(( subjectPhase[i][0], 
                            [mod_angle( s-t ) for s,t in zip(subjectPhase[i][1],templatePhase[i][1])] ))
        return dphase

    def retrain_gprmodel(self,**gpr_kwargs):
        """Train gprmodel again. This is usually necessary when the GPR class is modified and the performance
        values need to be calculated again.

        If available, use precoputed DTW with cost function to keep subject trajectory within
        bounds.

        Parameters
        ----------
        **gpr_kwargs
        """
        print("Retraining model...")
        from .coherence import DTWPerformance,GPREllipsoid
        perfEval=DTWPerformance(dt_threshold=.68)
        gprmodel=GPREllipsoid(tmin=self.gprmodel.tmin,tmax=self.gprmodel.tmax,
                              fmin=self.gprmodel.fmin,fmax=self.gprmodel.fmax,
                              mean_performance=self.gprmodel.performanceData.mean(),
                              **gpr_kwargs)
        p=np.zeros(len(self.timeSplitTrials['avatar']))
        
        # Try to load DTW alignment path that would have been calculated with regularization.
        version=self.person[-3:]
        homedr=os.path.expanduser('~')
        f=homedr+'/Dropbox/Research/tango/py/cache/dtw_v%s.p'%version
        if os.path.isfile(f):
            print("Using cached DTW path file.")
            pathList=pickle.load(open(f,'rb'))['pathList'][self._find_subject_settings_index()]
            assert len(self.templateSplitTrials['avatar'])==len(pathList)

            for i,(t,sv,avv,path) in enumerate(zip(self.timeSplitTrials['avatar'],
                                                   self.subjectSplitTrials['avatar'],
                                                   self.templateSplitTrials['avatar'],
                                                   pathList)):
                p[i]=perfEval.time_average_binary(avv[:,1:],sv[:,1:],
                                                  dt=1/30,#t[1]-t[0],
                                                  path=path)
        else:
            for i,(t,sv,avv) in enumerate(zip(self.timeSplitTrials['avatar'],
                                              self.subjectSplitTrials['avatar'],
                                              self.templateSplitTrials['avatar'])):
                p[i]=perfEval.time_average_binary(avv[:,1:],sv[:,1:],dt=t[1]-t[0],bds=[1,t.max()-1])
               
        assert ((1>p)&(p>0)).all()
        gprmodel.update(self.gprmodel.ilogistic(p),self.gprmodel.durations,self.gprmodel.fractions)
        self.gprmodel=gprmodel

    def _find_subject_settings_index(self):
        """Find where this subject and hand would be located in the flat list of all subject
        settings. This is bit hacked.

        Returns
        -------
        index : int
        """
        done=False

        if self.person[-3:]=='3_3':
            subject_settings=subject_settings_v3_3
        elif self.person[-3:]=='3_4':
            subject_settings=subject_settings_v3_4
        elif self.person[-3:]=='3_5':
            subject_settings=subject_settings_v3_5
        elif self.person[-3:]=='3_6':
            subject_settings=subject_settings_v3_6
        else:
            raise Exception("Unrecognized experiment version.")

        totalCounter=0
        subjectCounter=0
        while not done:
            try:
                person,dr,rotAngle,reverse,usable=subject_settings(subjectCounter,'left')
                if self.person==person and self.modelhandedness[0].lower()=='right':
                    return totalCounter
                elif usable:
                    totalCounter+=1

                person,dr,rotAngle,reverse,usable=subject_settings(subjectCounter,'right')
                if self.person==person and self.modelhandedness[0].lower()=='left':
                    return totalCounter
                elif usable:
                    totalCounter+=1
                subjectCounter+=1
            except IndexError:
                done=True
        
        raise Exception("Subject not found in list: %s"%self.person)

    def pickle_trial_dicts(self,disp=False):
        """
        Put data for analysis into easily accessible pickles. Right now, I extract only visibility and hand
        velocities for AN port data and avatar's motionbuilder files.
        
        Parameters
        ----------
        disp : bool,False
        """
        from .axis_neuron import extract_AN_port
        from .pipeline import extract_motionbuilder_model3_3
        from .utils import match_time
        from .ue4 import load_visibility
        import dill as pickle
        from .experiment import remove_pause_intervals

        # Load AN data.
        df = pickle.load(open('%s/%s'%(self.dr,'quickload_an_port_vr.p'),'rb'))['df']
        windowsByPart,_,_ = self.window_specs(self.person,self.dr)

        # Sort trials into the hand, arm, and avatar trial dictionaries: subjectTrial, templateTrial,
        # hmdTrials. These contain arrays for time that were interpolated in for regular sampling and
        # functions for velocities.
        subjectTrial,templateTrial,hmdTrials = {},{},{}
        timeSplitTrials,subjectSplitTrials,templateSplitTrials = {},{},{}

        for trialno,part in enumerate(self.trialTypes):
            if disp:
                print("Processing %s..."%part)

            # Load visibility time points saved by UE4 and remove pause intervals.
            if part.isalpha():
                visible,invisible = load_visibility(part+'_visibility',self.dr)
            else:
                visible,invisible = load_visibility(part[:-1]+'_visibility_0',self.dr)
            visible,_=remove_pause_intervals(visible.tolist(),list(zip(*self.pause)))
            invisible,_=remove_pause_intervals(invisible.tolist(),list(zip(*self.pause)))
            visible,invisible=np.array(visible),np.array(invisible)
            
            # Start and end times counting only the time the simulation is running (and not paused).
            exptStartEnd = [visible[0],invisible[-1]]
            
            # Extract template.
            mbV,mbT = extract_motionbuilder_model3_3( self.modelhandedness[trialno],
                                                      reverse_time=self.reverse )
            showIx = mbT < (exptStartEnd[1]-exptStartEnd[0]).total_seconds()
            templateTrial[part+'T'] = mbT[showIx][::2]
            templateTrial[part+'V'] = mbV
            
            # Extract subject from port file.
            anT,anX,anV,anA = extract_AN_port( df,self.modelhandedness[trialno],
                                               rotation_angle=self.rotation )
            # Remove pauses.
            anT,_,removeIx=remove_pause_intervals(anT.tolist(),list(zip(*self.pause)),True)
            anT=np.array(anT)
            anV=np.delete(anV[0],removeIx,axis=0)
            # Remove parts that extend beyond trial.
            showIx = (anT>=exptStartEnd[0]) & (anT<=exptStartEnd[1])
            anT,anV = anT[showIx],anV[showIx]
            # Save into variables used here.
            subjectTrial[part+'T'],subjectTrial[part+'V'] = anT,anV
            
            # Put trajectories on the same time samples so we can pipeline our regular computation.
            offset = (subjectTrial[part+'T'][0]-exptStartEnd[0]).total_seconds()
            subjectTrial[part+'V'],subjectTrial[part+'T'] = match_time(subjectTrial[part+'V'],
                                                                       subjectTrial[part+'T'],
                                                                       1/30,
                                                                       offset=offset,
                                                                       use_univariate=True)

            # Separate the different visible trials into separate arrays.
            # Times for when visible/invisible windows start.
            start = np.zeros((len(visible)+len(invisible)),dtype=object)
            start[::2] = visible
            start[1::2] = invisible
            # Units of seconds.
            start = np.array([t.total_seconds() for t in np.diff(start)])
            start = np.cumsum(start)
            invisibleStart = start[::2]  # as seconds
            visibleStart = start[1::2]  # as seconds
            
            # When target is invisible, set visibility to 0.
            visibility = np.ones_like(templateTrial[part+'T'])
            for i,j in zip(invisibleStart,visibleStart):
                if i<j:
                    visibility[(templateTrial[part+'T']>=i) & (templateTrial[part+'T']<j)] = 0
            if len(visible)<len(invisible):
                visibility[(templateTrial[part+'T']>=invisible[-1])] = 0
            templateTrial[part+'visibility'] = visibility
            
            # Separate single data take into separate trials.
            timeSplitTrials[part],subjectSplitTrials[part],templateSplitTrials[part] = [],[],[]
            templateSplitTrials[part+'visibility'] = []
            for spec,startendt in windowsByPart[part]:
                startendt = ((startendt[0]-exptStartEnd[0]).total_seconds(),
                             (startendt[1]-exptStartEnd[0]).total_seconds())

                # Save time.
                timeix = (templateTrial[part+'T']<=startendt[1])&(templateTrial[part+'T']>=startendt[0])
                t = templateTrial[part+'T'][timeix]
                timeSplitTrials[part].append(t)

                # Save visibility window.
                templateSplitTrials[part+'visibility'].append( visibility[timeix] )
                
                # Save velocities.
                templateSplitTrials[part].append( templateTrial[part+'V'](t) )
                subjectSplitTrials[part].append( subjectTrial[part+'V'](t) )
        
        pickle.dump({'templateTrial':templateTrial,
                     'subjectTrial':subjectTrial,
                     'timeSplitTrials':timeSplitTrials,
                     'templateSplitTrials':templateSplitTrials,
                     'subjectSplitTrials':subjectSplitTrials,
                     'windowsByPart':windowsByPart},
                    open('%s/trial_dictionaries.p'%self.dr,'wb'),-1)

    def _fetch_windowspec_indices(self,specs,trial_type,precision=None):
        """
        Given a particular trial type and a window specification, return all the indices within
        that trial type that match the given specification.  Options for adjusting the
        precision for matching windows.

        Parameters
        ----------
        trial_type : str
        spec : list of tuples
        
        Returns
        -------
        ix : list of ints
        """
        trialWindows = np.array([w[0] for w in self.windowsByPart[trial_type]])
        i = 0  # counter

        if precision is None:
            for spec in specs:
                ix_ = (np.array(spec)[None,:]==trialWindows).all(1)
                if ix_.any():
                    ix=np.where(ix_)[0].tolist()
                else:
                    ix=[]
        elif type(precision) is float:
            for spec in specs:
                specDiffs = np.abs( trialWindows-np.array(spec)[None,:] )
                ix_ = (specDiffs<=precision).all(1)
                if ix_.any():
                    ix=np.where(ix_)[0].tolist()
                else:
                    ix=[]
        elif type(precision) is tuple:
            for spec in specs:
                specDiffs = np.abs( trialWindows-np.array(spec)[None,:] )
                ix_ = (specDiffs[:,0]<=precision[0])&(specDiffs[:,1]<=precision[1])
                if ix_.any():
                    ix=np.where(ix_)[0].tolist()
                else:
                    ix=[]
        else:
            raise NotImplementedError("precision type not supported.")

        return ix

    def window_specs(self,person,dr,reload_trial_times=False):
        """
        Get when the different visible/invisible cycles occur in the given experiment. These data are
        obtained from visibility text files output from UE4.
        
        Parameters
        ----------
        person : str
            Will point to the folder that the data is in.
        dr : str
        reload_trial_times : bool,False
            If True, identify trial start and end times from the visibility data.

        Returns
        -------
        windowsByPart : dict
            Keys correspond to trial types. Each dict entry is a list of tuples ((type of
            window),(window start, window end)) Window type is a tuple
            (inv_duration,window_duration)
        """
        from .ue4 import load_visibility 

        # Load AN subject data.
        df = pickle.load(open('%s/%s'%(dr,'quickload_an_port_vr.p'),'r'))['df']

        windowsByPart = {}
        
        for trialno,part in enumerate(['avatar']):
            if part.isalpha():
                fname = part+'_visibility'
            else:
                fname = part[:-1]+'_visibility_0'

            visible,invisible = load_visibility(fname,dr)
            visible=self._remove_pauses(visible)
            invisible=self._remove_pauses(invisible)

            # Array denoting visible (with 1) and invisible (with 0) times.
            start = np.zeros((len(visible)+len(invisible)),dtype=object)
            start[::2] = visible
            start[1::2] = invisible
            start = np.array([t.total_seconds() for t in np.diff(start)])
            start = np.cumsum(start)
            invisibleStart = start[::2]
            visibleStart = start[1::2]

            # Load data saved in gpr.p.
            # The first time point is when the file was written which we can throw out. The second pair of
            # times are when the trial counter is updated immediately after the first fully visible trial. The
            # remaining points are the following trials.
            dataDict = pickle.load(open('%s/%s'%(self.dr,'gpr.p'),'rb'))
            if reload_trial_times:
                t0,t1=infer_trial_times_from_visibility(self.pause[0],self.pause[1],self.dr)
                trialStartTimes,trialEndTimes=t0,t1
            else:
                trialStartTimes = self._remove_pauses(dataDict['trialStartTimes'])
                trialEndTimes = self._remove_pauses(dataDict['trialEndTimes'])
            windowSpecs = []
            windowStart,windowEnd = [],[]
            for i in range(len(self.gprmodel.fractions)):
                if i==0:
                    windowSpecs.append((0,0))
                    windowStart.append(visible[0])
                    windowEnd.append(trialStartTimes[1])
                else:
                    invDur = (1-self.gprmodel.fractions[i])*self.gprmodel.durations[i]
                    winDur = self.gprmodel.durations[i]
                    windowSpecs.append((invDur,winDur))

                    windowStart.append(trialStartTimes[i+1])
                    windowEnd.append(trialEndTimes[i+1])

            windowsByPart[part] = list(zip(windowSpecs,list(zip(windowStart,windowEnd))))

            # Get the duration of the invisible and visible windows in the time series.
            mxLen = min([len(visibleStart),len(invisibleStart)])
            invDur = visibleStart[:mxLen]-invisibleStart[:mxLen]
            visDur = invisibleStart[1:][:mxLen-1]-visibleStart[:-1][:mxLen-1]
            windowDur = invDur[:-1]+visDur  # total duration cycle of visible and invisible
        return windowsByPart,invDur,visDur

    def _remove_pauses(self,x):
        """Wrapper around experiment.remove_pause_intervals."""
        from .experiment import remove_pause_intervals
        if type(x) is list:
            x,_=remove_pause_intervals(x,list(zip(*self.pause)))
            return x
        x,_=remove_pause_intervals(x.tolist(),list(zip(*self.pause)))
        return np.array(x)
#end VRTrial3_1



class BuggyVRTrial3_5(VRTrial3_1):
    #def __init__(self):
    #    super(BuggyVRTrial3_5,self).__init__()
    #    if self.person=='Subject13_3_5':
    #        self.remove_first_trial()

    #def remove_first_trial(self):
    #    self.timeSplitTrials=self.timeSplitTrials['avatar'][1:]
    #    self.subjectSplitTrials=self.subjectSplitTrials['avatar'][1:]
    #    self.templateSplitTrials=self.templateSplitTrials['avatar'][1:]
    #    self.windowsByPart=self.windowsByPart['avatar'][1:]

    def retrain_gprmodel(self,**gpr_kwargs):
        """Train gprmodel again. This is usually necessary when the GPR class is modified and the performance
        values need to be calculated again.

        If available, use precoputed DTW with cost function to keep subject trajectory within
        bounds.

        Parameters
        ----------
        **gpr_kwargs
        """
        print("Retraining model...")
        from .coherence import DTWPerformance,GPREllipsoid
        perfEval=DTWPerformance(dt_threshold=.68)
        gprmodel=GPREllipsoid(tmin=self.gprmodel.tmin,tmax=self.gprmodel.tmax,
                              fmin=self.gprmodel.fmin,fmax=self.gprmodel.fmax,
                              mean_performance=self.gprmodel.performanceData.mean(),
                              **gpr_kwargs)
        p=np.zeros(len(self.timeSplitTrials['avatar']))
        
        # Try to load DTW alignment path that would have been calculated with regularization.
        version=self.person[-3:]
        homedr=os.path.expanduser('~')
        f=homedr+'/Dropbox/Research/tango/py/cache/dtw_v%s.p'%version
        frac,dur=[],[]
        if os.path.isfile(f):
            print("Using cached DTW path file.")
            pathList=pickle.load(open(f,'rb'))['pathList'][self._find_subject_settings_index()]
            assert len(self.templateSplitTrials['avatar'])==len(pathList)

            for i,(t,sv,avv,(windowSpec,_),path) in enumerate(zip(self.timeSplitTrials['avatar'],
                                                                  self.subjectSplitTrials['avatar'],
                                                                  self.templateSplitTrials['avatar'],
                                                                  self.windowsByPart['avatar'],
                                                                  pathList)):
                p[i]=perfEval.time_average_binary(avv[:,1:],sv[:,1:],
                                                  dt=1/30,#t[1]-t[0],
                                                  path=path)
                if windowSpec[1]==0:
                    frac.append(1.)
                    dur.append(0.)
                else:
                    frac.append( (windowSpec[1]-windowSpec[0])/windowSpec[1] )
                    dur.append( windowSpec[1] )

        else:
            for i,(t,sv,avv,(windowSpec,_)) in enumerate(zip(self.timeSplitTrials['avatar'],
                                                         self.subjectSplitTrials['avatar'],
                                                         self.templateSplitTrials['avatar'],
                                                         self.windowsByPart['avatar'])):

                p[i]=perfEval.time_average_binary(avv[:,1:],sv[:,1:],dt=t[1]-t[0],bds=[1,t.max()-1])

                if windowSpec[1]==0:
                    frac.append(1.)
                    dur.append(0.)
                else:
                    frac.append( (windowSpec[1]-windowSpec[0])/windowSpec[1] )
                    dur.append( windowSpec[1] )

        assert ((1>p)&(p>0)).all()
        gprmodel.update( self.gprmodel.ilogistic(p),np.array(dur),np.array(frac) )
        self.gprmodel=gprmodel

    def pickle_trial_dicts(self,disp=False):
        """
        Put data for analysis into easily accessible pickles. Right now, I extract only visibility and hand
        velocities for AN port data and avatar's motionbuilder files.
        
        Parameters
        ----------
        disp : bool,False
        """
        from .axis_neuron import extract_AN_port
        from .pipeline import extract_motionbuilder_model3_3
        from .utils import match_time
        from .ue4 import load_visibility
        import dill as pickle
        from .experiment import remove_pause_intervals

        # Load AN data.
        df = pickle.load(open('%s/%s'%(self.dr,'quickload_an_port_vr.p'),'rb'))['df']
        windowsByPart,_,_ = self.window_specs(self.person,self.dr)

        # Sort trials into the hand, arm, and avatar trial dictionaries: subjectTrial, templateTrial,
        # hmdTrials. These contain arrays for time that were interpolated in for regular sampling and
        # functions for velocities.
        subjectTrial,templateTrial,hmdTrials = {},{},{}
        timeSplitTrials,subjectSplitTrials,templateSplitTrials = {},{},{}

        for trialno,part in enumerate(self.trialTypes):
            if disp:
                print("Processing %s..."%part)

            # Load visibility time points saved by UE4 and remove pause intervals.
            if part.isalpha():
                visible,invisible = load_visibility(part+'_visibility',self.dr)
            else:
                visible,invisible = load_visibility(part[:-1]+'_visibility_0',self.dr)
            visible,_=remove_pause_intervals(visible.tolist(),list(zip(*self.pause)))
            invisible,_=remove_pause_intervals(invisible.tolist(),list(zip(*self.pause)))
            visible,invisible=np.array(visible),np.array(invisible)
            
            # Start and end times counting only the time the simulation is running (and not paused).
            exptStartEnd = [visible[0],invisible[-1]]
            
            # Extract template. Downsample to 30Hz from 60Hz.
            mbV,mbT = extract_motionbuilder_model3_3( self.modelhandedness[trialno],
                                                      reverse_time=self.reverse )
            showIx = mbT < (exptStartEnd[1]-exptStartEnd[0]).total_seconds()
            templateTrial[part+'T'] = mbT[showIx][::2]
            templateTrial[part+'V'] = mbV
            
            # Extract subject from port file.
            anT,anX,anV,anA = extract_AN_port( df,self.modelhandedness[trialno],
                                               rotation_angle=self.rotation )
            # Remove pauses.
            anT,_,removeIx=remove_pause_intervals(anT.tolist(),list(zip(*self.pause)),True)
            anT=np.array(anT)
            anV=np.delete(anV[0],removeIx,axis=0)
            # Remove parts that extend beyond trial.
            showIx = (anT>=exptStartEnd[0]) & (anT<=exptStartEnd[1])
            anT,anV = anT[showIx],anV[showIx]
            # Save into variables used here.
            subjectTrial[part+'T'],subjectTrial[part+'V'] = anT,anV
            
            # Put trajectories on the same time samples so we can pipeline our regular computation.
            offset = (subjectTrial[part+'T'][0]-exptStartEnd[0]).total_seconds()
            subjectTrial[part+'V'],subjectTrial[part+'T'] = match_time(subjectTrial[part+'V'],
                                                                       subjectTrial[part+'T'],
                                                                       1/30,
                                                                       offset=offset,
                                                                       use_univariate=True)

            # Separate the different visible trials into separate arrays.
            # Times for when visible/invisible windows start.
            start = np.zeros((len(visible)+len(invisible)),dtype=object)
            start[::2] = visible
            start[1::2] = invisible
            # Units of seconds.
            start = np.array([t.total_seconds() for t in np.diff(start)])
            start = np.cumsum(start)
            invisibleStart = start[::2]  # as seconds
            visibleStart = start[1::2]  # as seconds
            
            # When target is invisible, set visibility to 0.
            visibility = np.ones_like(templateTrial[part+'T'])
            for i,j in zip(invisibleStart,visibleStart):
                if i<j:
                    visibility[(templateTrial[part+'T']>=i) & (templateTrial[part+'T']<j)] = 0
            if len(visible)<len(invisible):
                visibility[(templateTrial[part+'T']>=invisible[-1])] = 0
            templateTrial[part+'visibility'] = visibility
            
            # Separate single data take into separate trials.
            timeSplitTrials[part],subjectSplitTrials[part],templateSplitTrials[part] = [],[],[]
            templateSplitTrials[part+'visibility'] = []
            for spec,startendt in windowsByPart[part]:
                startendt = ((startendt[0]-exptStartEnd[0]).total_seconds(),
                             (startendt[1]-exptStartEnd[0]).total_seconds())

                # Save time.
                timeix = (templateTrial[part+'T']<=startendt[1])&(templateTrial[part+'T']>=startendt[0])
                t = templateTrial[part+'T'][timeix]
                timeSplitTrials[part].append(t)

                # Save visibility window.
                templateSplitTrials[part+'visibility'].append( visibility[timeix] )
                
                # Save velocities.
                templateSplitTrials[part].append( templateTrial[part+'V'](t) )
                subjectSplitTrials[part].append( subjectTrial[part+'V'](t) )
        
        if self.person=='Subject13_3_5':
            timeSplitTrials['avatar']=timeSplitTrials['avatar'][1:]
            subjectSplitTrials['avatar']=subjectSplitTrials['avatar'][1:]
            templateSplitTrials['avatar']=templateSplitTrials['avatar'][1:]
            templateSplitTrials['avatarvisibility']=templateSplitTrials['avatarvisibility'][1:]
            windowsByPart['avatar']=windowsByPart['avatar'][1:]

        pickle.dump({'templateTrial':templateTrial,
                     'subjectTrial':subjectTrial,
                     'timeSplitTrials':timeSplitTrials,
                     'templateSplitTrials':templateSplitTrials,
                     'subjectSplitTrials':subjectSplitTrials,
                     'windowsByPart':windowsByPart},
                    open('%s/trial_dictionaries.p'%self.dr,'wb'),-1)

    def window_specs(self,person,dr):
        """
        Get when the different visible/invisible cycles occur in the given experiment. These data are
        obtained from visibility text files output from UE4.
        
        Parameters
        ----------
        person : str
            Will point to the folder that the data is in.
        dr : str

        Returns
        -------
        windowsByPart : dict
            Keys correspond to trial types. Each dict entry is a list of tuples ((type of
            window),(window start, window end)) Window type is a tuple
            (inv_duration,window_duration)
        """
        from .ue4 import load_visibility 

        # Load AN subject data.
        df = pickle.load(open('%s/%s'%(dr,'quickload_an_port_vr.p'),'r'))['df']

        windowsByPart = {}
        
        for trialno,part in enumerate(['avatar']):
            if part.isalpha():
                fname = part+'_visibility'
            else:
                fname = part[:-1]+'_visibility_0'

            visible,invisible = load_visibility(fname,dr)
            visible=self._remove_pauses(visible)
            invisible=self._remove_pauses(invisible)

            # Array denoting visible (with 1) and invisible (with 0) times.
            start = np.zeros((len(visible)+len(invisible)),dtype=object)
            start[::2] = visible
            start[1::2] = invisible
            start = np.array([t.total_seconds() for t in np.diff(start)])
            start = np.cumsum(start)
            invisibleStart = start[::2]
            visibleStart = start[1::2]

            # Load data saved in gpr.p.
            # The first time point is when the file was written which we can throw out. The second pair of
            # times are when the trial counter is updated immediately after the first fully visible trial. The
            # remaining points are the following trials.
            dataDict = pickle.load(open('%s/%s'%(self.dr,'gpr.p'),'rb'))
            if self.person=='Subject01_3_5':
                # Exception for anomalous trial.
                t0,t1,invDur,windowDur = infer_trial_times_from_visibility(self.pause[0],self.pause[1],self.dr,
                                                                           visible_duration=15)
            else:
                t0,t1,invDur,windowDur=infer_trial_times_from_visibility(self.pause[0],self.pause[1],self.dr)
            trialStartTimes,trialEndTimes=t0,t1

            windowSpecs = []
            windowStart,windowEnd = [],[]
            for i in range(len(trialStartTimes)):
                if i==0:
                    windowSpecs.append((0,0))
                else:
                    windowSpecs.append((invDur[i],windowDur[i]))

                windowStart.append(trialStartTimes[i])
                windowEnd.append(trialEndTimes[i])

            windowsByPart[part] = list(zip(windowSpecs,list(zip(windowStart,windowEnd))))

            # Get the duration of the invisible and visible windows in the time series.
            mxLen = min([len(visibleStart),len(invisibleStart)])
            invDur = visibleStart[:mxLen]-invisibleStart[:mxLen]
            visDur = invisibleStart[1:][:mxLen-1]-visibleStart[:-1][:mxLen-1]
            #windowDur = invDur[:-1]+visDur  # total duration cycle of visible and invisible
        return windowsByPart,invDur,visDur
#end BuggyVRTrial3_5



class VRTrial3(object):
    def __init__(self,person,modelhandedness,rotation,dr,fname='trial_dictionaries.p'):
        """
        Parameters
        ----------
        person (str)
        modelhandedness (list of str)
        rotation (list of float)
        dr (str)

        Attributes
        ----------
        person
        modelhandedness
        rotation
        dr
        subjectTrial (dict)
            Full Axis Neuron trial data labeled by part+'T' part+'V'.
        templateTrial (dict)
            Full MotionBuilder trial data labeled by part+'T' part+'V'.
        timeSplitTrials
        subjectSplitTrials
        templateSplitTrials

        Methods
        -------
        info
        subject_by_window_dur
        subject_by_window_spec
        pickle_trial_dicts
        pickle_phase
        _fetch_windowspec_indices
        """
        self.person = person
        self.modelhandedness = modelhandedness
        self.rotation = rotation
        self.dr = dr
        
        try:
            data = pickle.load(open('%s/%s'%(self.dr,fname),'rb'))
        except Exception:
            self.pickle_trial_dicts(1)
            
        data = pickle.load(open('%s/%s'%(self.dr,fname),'rb'))
        self.templateTrial = data['templateTrial']
        self.subjectTrial = data['subjectTrial']
        self.timeSplitTrials = data['timeSplitTrials']
        self.templateSplitTrials = data['templateSplitTrials']
        self.subjectSplitTrials = data['subjectSplitTrials']
        self.windowsByPart = data['windowsByPart']

    def info(self):
        print("Person %s"%self.person)
        print("Trials available:")
        for part in ['avatar','avatar0','hand','hand0']:
            print("%s\tInvisible\tTotal"%part)
            for spec,_ in self.windowsByPart[part]:
                print("\t%1.2f\t\t%1.2f"%(spec[0],spec[1]))
    
    def subject_by_window_dur(self,windowDur,part):
        """
        Params:
        -------
        windowDur (list)
            Duration of visible/invisible cycle.
        part (str)
            Body part to return.
            
        Returns:
        --------
        selection (list)
            List of trials that have given window duration. Each tuple in list is a tuple of the 
            ( (invisible,total window), time, extracted velocity data ).
        """
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.subjectSplitTrials[part][i] ))
        return selection
    
    def template_by_window_dur(self,windowDur,part):
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.templateSplitTrials[part][i] ))
        return selection
    
    def subject_by_invisible_dur(self,windowDur,part):
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[0]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.subjectSplitTrials[part][i] ))
        return selection

    def subject_by_window_spec(self,windowSpec,trial_type,precision=None):
        """Automatically search through left and right hand trials."""
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.subjectSplitTrials[trial_type][ix[0]] ))
            # Iterate also through hand0 or avatar0, which contains the other hand..
            if trial_type.isalpha():
                selection += self.subject_by_window_spec([windowSpec[specix]],
                                                             trial_type+'0',
                                                             precision=precision)
        return selection

    def template_by_window_spec(self,windowSpec,trial_type,precision=None):
        """Automatically search through left and right hand trials."""
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.templateSplitTrials[trial_type][ix[0]] ))
            # Iterate also through hand0 or avatar0, which contains the other hand..
            if trial_type.isalpha():
                selection += self.template_by_window_spec([windowSpec[specix]],
                                                             trial_type+'0',
                                                             precision=precision)
        return selection

    def template_by_invisible_dur(self,windowDur,part):
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[0]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.templateSplitTrials[part][i] ))
        if trialType.isalpha():
            return selection + self.template_by_invisible_dur(windowSpec,trialType+'0')
        return selection

    def visibility_by_window_spec(self,windowSpec,trial_type,precision=None):
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.templateSplitTrials[trial_type+'visibility'][ix[0]] ))
            # Iterate also through hand0 or avatar0, which contains the other hand..
            if trial_type.isalpha():
                selection += self.visibility_by_window_spec([windowSpec[specix]],
                                                             trial_type+'0',
                                                             precision=precision)
        return selection

    def phase_by_window_dur(self,source,windowDur,trialType):
        """
        Return instantaneous phase from bandpass filtered velocities on trial specificied by window
        duration.

        Params:
        -------
        source (str)
        windowDur (list of floats)
        trialType (str)
            'avatar', 'avatar0', 'hand', 'hand0'
        """
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i = 0
        for spec,_ in self.windowsByPart[trialType]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            try:
                if source=='subject' or source=='s':
                    phases = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trialType,i),
                                         'rb'))['phases']
                elif source=='template' or source=='t':
                    phases = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trialType,i),
                                         'rb'))['phases']

                phases = [np.vstack(p) for p in phases]
                selection.append(( self.windowsByPart[trialType][i][0],phases ))
            except IOError:
                print("Trial %d in trial type %s not found."%(i,trialType))
        return selection

    def phase_by_window_spec(self,source,windowSpec,trial_type):
        """
        Parameters
        ----------
        source : str
        windowSpec : list
        trial_type : str
        """
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.templateSplitTrials[trial_type][ix[0]] ))
                try:
                    if source=='subject' or source=='s':
                        data = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trial_type,ix[0]),'rb'))
                        phases,vs = data['phases'],data['vs']
                    elif source=='template' or source=='t':
                        data = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trial_type,ix[0]),'rb'))
                        phases,vs = data['phases'],data['vs']
                    
                    phases = [np.vstack(p) for p in phases]
                    selection.append(( self.windowsByPart[trial_type][ix[0]][0],phases ))
                except IOError:
                    print("Trial %d in trial type %s not found."%(ix[0],trial_type))

            # Iterate also through hand0 or avatar0, which contains the other hand.
            if trial_type.isalpha():
                selection += self.phase_by_window_spec(source,
                                                        [windowSpec[specix]],
                                                        trial_type+'0',
                                                        precision=precision)
        return selection

    def filtv_by_window_spec(self,source,windowSpec,trialType,search_all=True):
        """
        Returns:
        --------
        list of twoples (windowSpec, filtv) where filtv is a list of 3 arrays corresponding to each dimension
        """
        raise NotImplementedError()
        ix = self._fetch_windowspec_indices(windowSpec,trialType,precision=precision)
        selection = []

        for i in ix:
            if source=='subject' or source=='s':
                data = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))
                phases,vs = data['phases'],data['vs']
            elif source=='template' or source=='t':
                data = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))
                phases,vs = data['phases'],data['vs']
            else:
                raise Exception

            vs = [np.vstack(p) for p in vs]
            selection.append(( self.windowsByPart[trialType][i][0],vs ))

        if trialType.isalpha() and search_all:
            return selection + self.filtv_by_window_spec(source,windowSpec,trialType+'0',False)

        return selection

    def dphase_by_window_dur(self,windowDur,trialType):
        """
        Difference in phase between subject and template motion.
        """
        raise NotImplementedError
        from misc.angle import mod_angle
        
        subjectPhase = self.phase_by_window_dur('s',windowDur,trialType)
        templatePhase = self.phase_by_window_dur('t',windowDur,trialType)
        dphase = []
        
        for i in range(len(subjectPhase)):
            dphase.append(( subjectPhase[i][0], 
                            [mod_angle( s-t ) for s,t in zip(subjectPhase[i][1],templatePhase[i][1])] ))
        if trialType.isalpha():
            return dphase + self.dphase_by_window_dur(windowDur,trialType+'0')
        return dphase

    def dphase_by_window_spec(self,windowSpec,trialType):
        """
        Difference in phase between subject and template motion.
        """
        from misc.angle import mod_angle
        
        subjectPhase = self.phase_by_window_spec('s',windowSpec,trialType)
        templatePhase = self.phase_by_window_spec('t',windowSpec,trialType)
        dphase = []
            
        for i in range(len(subjectPhase)):
            dphase.append(( subjectPhase[i][0], 
                            [mod_angle( s-t ) for s,t in zip(subjectPhase[i][1],templatePhase[i][1])] ))
        return dphase

    def pickle_trial_dicts(self,disp=False):
        """
        Put data for analysis into easily accessible pickles. Right now, I extract only visibility
        and hand velocities for AN port data and avatar's motionbuilder files.
        """
        from .pipeline import extract_motionbuilder_model2,extract_AN_port
        from .utils import match_time

        # Load AN data.
        df = pickle.load(open('%s/%s'%(self.dr,'quickload_an_port_vr.p'),'rb'))['df']
        windowsByPart = window_specs(self.person,self.dr)
        
        # Sort trials into the hand, arm, and avatar trial dictionaries: subjectTrial,
        # templateTrial, hmdTrials.
        subjectTrial,templateTrial,hmdTrials = {},{},{}
        timeSplitTrials,subjectSplitTrials,templateSplitTrials = {},{},{}

        for trialno,part in enumerate(['avatar','avatar0','hand','hand0']):
            if disp:
                print("Processing %s..."%part)
            # Select time interval during which the trial happened.
            if part.isalpha():
                visible,invisible = load_visibility(part+'_visibility.txt',self.dr)
            else:
                visible,invisible = load_visibility(part[:-1]+'_visibility_0.txt',self.dr)
            startEnd = [visible[0],visible[-1]]
            
            # Extract template.
            mbT,mbV = extract_motionbuilder_model2(part,startEnd[0],self.modelhandedness[trialno])
            showIx = (mbT>startEnd[0]) & (mbT<startEnd[1])
            templateTrial[part+'T'],templateTrial[part+'V'] = mbT[showIx],mbV[showIx]
            
            # Extract subject from port file.
            anT,anX,anV,anA = extract_AN_port( df,self.modelhandedness[trialno],
                                               rotation_angle=self.rotation[trialno] )
            showIx = (anT>startEnd[0]) & (anT<startEnd[1])
            subjectTrial[part+'T'],subjectTrial[part+'V'] = anT[showIx],anV[0][showIx]
            
            if disp:
                print(("For trial %s, template ends at %s and subject at "+
                        "%s.")%(part,
                                str(templateTrial[part+'T'][-1])[11:],
                                str(subjectTrial[part+'T'][-1])[11:]))

            # Put trajectories on the same time samples so we can pipeline our regular computation.
            # Since the AN trial starts after the mbTrial...the offset is positive.
            subjectTrial[part+'V'],subjectTrial[part+'T'] = match_time(subjectTrial[part+'V'],
                   subjectTrial[part+'T'],
                   1/60,
                   offset=(subjectTrial[part+'T'][0]-templateTrial[part+'T'][0]).total_seconds(),
                   use_univariate=True)
            templateTrial[part+'V'],templateTrial[part+'T'] = match_time(templateTrial[part+'V'],
                                                                     templateTrial[part+'T'],
                                                                     1/60,
                                                                     use_univariate=True)
            
            # Separate the different visible trials into separate arrays.
            # Times for when visible/invisible windows start.
            start = np.zeros((len(visible)+len(invisible)),dtype=object)
            start[::2] = visible
            start[1::2] = invisible
            # Units of seconds.
            start = np.array([t.total_seconds() for t in np.diff(start)])
            start = np.cumsum(start)
            invisibleStart = start[::2]  # as seconds
            visibleStart = start[1::2]  # as seconds
            
            # When target is invisible, set visibility to 0.
            visibility = np.ones_like(templateTrial[part+'T'])
            for i,j in zip(invisibleStart,visibleStart):
                assert i<j
                visibility[(templateTrial[part+'T']>=i) & (templateTrial[part+'T']<j)] = 0
            if len(visible)<len(invisible):
                visibility[(templateTrial[part+'T']>=invisible[-1])] = 0
            templateTrial[part+'visibility'] = visibility

            timeSplitTrials[part],subjectSplitTrials[part],templateSplitTrials[part] = [],[],[]
            templateSplitTrials[part+'visibility'] = []
            for spec,startendt in windowsByPart[part]:
                startendt = ((startendt[0]-startEnd[0]).total_seconds(),
                             (startendt[1]-startEnd[0]).total_seconds())

                # Save time.
                timeix = (templateTrial[part+'T']<=startendt[1])&(templateTrial[part+'T']>=startendt[0])
                t = templateTrial[part+'T'][timeix]
                timeSplitTrials[part].append(t)

                # Save visibility window.
                templateSplitTrials[part+'visibility'].append( visibility[timeix] )
                
                # Save velocities.
                templateSplitTrials[part].append( templateTrial[part+'V'](t) )
                # Subject sometimes has cutoff window so must reindex time.
                timeix = (subjectTrial[part+'T']<=startendt[1])&(subjectTrial[part+'T']>=startendt[0])
                t = subjectTrial[part+'T'][timeix]
                subjectSplitTrials[part].append( subjectTrial[part+'V'](t) )
            
            # Get the beginning fully visible window. Insert this into the beginning of the list.
            windowsByPart[part].insert(0,((0,0),(0,invisibleStart[0])))
            timeix = (subjectTrial[part+'T']>=0)&(subjectTrial[part+'T']<=invisibleStart[0])
            t = subjectTrial[part+'T'][timeix]
            
            timeSplitTrials[part].insert(0,t)
            subjectSplitTrials[part].insert( 0,subjectTrial[part+'V'](t) )
            templateSplitTrials[part].insert( 0,templateTrial[part+'V'](t) )

            timeix = (templateTrial[part+'T']<=invisibleStart[0])&(templateTrial[part+'T']>=0)
            templateSplitTrials[part+'visibility'].insert( 0,visibility[timeix] )
        
        pickle.dump({'templateTrial':templateTrial,
                     'subjectTrial':subjectTrial,
                     'timeSplitTrials':timeSplitTrials,
                     'templateSplitTrials':templateSplitTrials,
                     'subjectSplitTrials':subjectSplitTrials,
                     'windowsByPart':windowsByPart},
                    open('%s/trial_dictionaries.p'%self.dr,'wb'),-1)

    def pickle_phase(self,trial_types=['avatar','avatar0','hand','hand0']):
        """
        Calculate bandpass filtered phase and pickle.
        """
        from .pipeline import pipeline_phase_calc
        
        for part in trial_types:
            nTrials = len(self.windowsByPart[part])  # number of trials for that part

            # Subject.
            toProcess = []
            trialNumbers = []
            for i in range(nTrials):
                # Only run process if we have data points. Some trials are missing data points.
                # NOTE: At some point the min length should made to correspond to the min window
                # size in the windowing function for filtering.
                if len(self.timeSplitTrials[part][i])>501:
                    trialNumbers.append(i)
                    toProcess.append( (self.timeSplitTrials[part][i],
                                       (self.subjectSplitTrials[part][i][:,0],
                                        self.subjectSplitTrials[part][i][:,1],
                                        self.subjectSplitTrials[part][i][:,2])) )
                else:
                    print("Ignoring %s trial no %d with windowspec (%1.1f,%1.1f)."%(part,i,
                        self.windowsByPart[part][i][0][0],self.windowsByPart[part][i][0][1]))
            pipeline_phase_calc(trajs=toProcess,dr=self.dr,
                                file_names=['subject_phase_%s_%d'%(part,i)
                                            for i in trialNumbers])
            # Template.
            toProcess = []
            trialNumbers = []
            for i in range(nTrials):
                if len(self.timeSplitTrials[part][i])>501:
                    trialNumbers.append(i)
                    toProcess.append( (self.timeSplitTrials[part][i],
                                       (self.templateSplitTrials[part][i][:,0],
                                        self.templateSplitTrials[part][i][:,1],
                                        self.templateSplitTrials[part][i][:,2])) )
            pipeline_phase_calc(trajs=toProcess,dr=self.dr,
                                file_names=['template_phase_%s_%d'%(part,i)
                                            for i in trialNumbers])

    def _fetch_windowspec_indices(self,specs,trial_type,precision=None):
        """
        Given a particular trial type and a window specification, return all the indices within that
        trial type that match the given specification.  Options for adjusting the precision for
        matching windows.

        Params
        ------
        trial_type : str
        spec : list of tuples
            Each twople is (duration_invisible,window_duration).
        
        Returns
        -------
        ix : list of ints
        """
        ix = []
        trialWindows = np.array([w[0] for w in self.windowsByPart[trial_type]])
        i = 0  # counter

        if precision is None:
            for spec in specs:
                ix_ = np.isclose(np.array(spec)[None,:],trialWindows).all(1)
                if ix_.any():
                    ix.append( np.where(ix_)[0][0] )
        elif type(precision) is float:
            for spec in specs:
                specDiffs = np.abs( trialWindows-np.array(spec)[None,:] )
                ix_ = (specDiffs<=precision).all(1)
                if ix_.any():
                    ix.append(np.where(ix_)[0][0])
        elif type(precision) is tuple:
            for spec in specs:
                specDiffs = np.abs( trialWindows-np.array(spec)[None,:] )
                ix_ = (specDiffs[:,0]<=precision[0])&(specDiffs[:,1]<=precision[1])
                if ix_.any():
                    ix.append(np.where(ix_)[0][0])
        else:
            raise NotImplementedError("precision type not supported.")

        return ix
# end VRTrial3


class Node(object):
    def __init__(self,name=None,parents=[],children=[]):
        self.name = name
        self.parents = parents
        self.children = children

    def add_child(self,child):
        self.children.append(child)

class Tree(object):
    def __init__(self,nodes):
        """
        Data structure for BVH skeleton hierarchy.

        Attributes:
        -----------
        _nodes (Node)
        nodes
        adjacency
        """
        self._nodes = nodes
        self.nodes = [n.name for n in nodes]
        names = [n.name for n in nodes]
        if len(np.unique(names))<len(names):
            raise Exception("Nodes have duplicate names.")

        self.adjacency = np.zeros((len(nodes),len(nodes)))
        for i,n in enumerate(nodes):
            for c in n.children:
                try:
                    self.adjacency[i,names.index(c)] = 1
                # automatically insert missing nodes (these should all be dangling)
                except ValueError:
                    self.adjacency = np.pad( self.adjacency, ((0,1),(0,1)), mode='constant', constant_values=0)
                    self._nodes.append( Node(c) )
                    names.append(c)

                    self.adjacency[i,names.index(c)] = 1
        
    def print_tree(self):
        print(self.adjacency)
    
    def parents(self,node):
        """
        Return parents of particular node.

        Returns:
        --------
        parents (list)
            Parents starting from immediate parent and ascending up the tree.
        """
        parents = []
        iloc = self.nodes.index(node)

        while np.any(self.adjacency[:,iloc]):
            iloc = np.where(self.adjacency[:,iloc])[0][0]
            parents.append(self.nodes[iloc])

        return parents



def infer_trial_times_from_visibility(pause,unpause,dr,
                                      file_name='avatar_visibility',
                                      visible_duration=30):
    """Identify endpoints of trials from the visibility/invisibility times. This will 
    return the start and end times for 16 trials that are about 30 s long.
    
    Parameters
    ----------
    pause : list of datetime
        Usually there are only three pauses.
    unpause : list datetime
    dr : str
    file_name : str,'avatar_visibility'
    visible_duration : float,30
        Duration of visible trials.
    
    Returns
    -------
    trialStartTimes : list
    trialEndTimes : list
    invDur : list
        Invisible duration for this set of trials as pulled from the first window in this trial.
    windowDur : list
        Total window duration for this set of trials as pulled from the first window in this trial.
    """
    from datetime import timedelta
    from perceptionneuron.ue4 import load_visibility
    from perceptionneuron.experiment import remove_pause_intervals

    visible,invisible=load_visibility('%s/%s'%(dr,'avatar_visibility'))
    visible,_=remove_pause_intervals(visible,list(zip(pause,unpause)))
    invisible,_=remove_pause_intervals(invisible,list(zip(pause,unpause)))
    visible=visible.tolist()
    invisible=invisible.tolist()

    # Duration of each visibility cycle.
    dt=[i.total_seconds() 
        for i in np.diff( np.vstack(list(zip(visible,invisible))),1 ).ravel()]
    dt2=[i.total_seconds() 
         for i in np.diff( np.vstack(list(zip(invisible[:-1],visible[1:]))),1 ).ravel()]
    assert (np.array(dt)>0).all()
    if not (np.around(dt[0])==30 and np.around(dt[-1])==30, (dt[0],dt[-1])):
        msg="Initial and final trials are not 30s: %1.2f and %1.2f."%(dt[0],dt[-1])
        warn(msg)

    # Loop through all the visibility windows and identify when they change.
    # First trial following full visibility window is a special case.
    trialStartTimes=[visible.pop(0)]
    trialEndTimes=[invisible.pop(0)]
    invDur=[0.]
    windowDur=[0.]
    dt.pop(0)
    dt2.pop(0)

    lastdt=nowdt=dt.pop(0)
    lastdt2=nowdt2=dt2.pop(0)
    while len(dt2)>0:
        if len(dt2)==1:
            # Case where second to last change can be a weird blip.
            invisible.pop(0)
        elif (abs(nowdt-lastdt)+abs(nowdt2-lastdt2)+
              abs(nowdt+nowdt2-lastdt-lastdt2))>.1:
            trialStartTimes.append(trialEndTimes[-1])
            trialEndTimes.append(invisible.pop(0))
            invDur.append(lastdt2)
            windowDur.append(lastdt2+lastdt)
        else:
            invisible.pop(0)
        visible.pop(0)
        lastdt=nowdt
        lastdt2=nowdt2
        nowdt=dt.pop(0)
        nowdt2=dt2.pop(0)
    trialStartTimes.append(trialEndTimes[-1])
    trialEndTimes.append(invisible.pop(0))
    invDur.append(lastdt2)
    windowDur.append(lastdt2+lastdt)
    #Last visible trial.
    trialStartTimes.append(visible.pop(0))
    trialEndTimes.append(trialStartTimes[-1]+timedelta(seconds=visible_duration))
    invDur.append(0.)
    windowDur.append(0.)
    
    # Checks.
    assert len(trialStartTimes)==len(trialEndTimes)
    
    # Only keep trials that are more than 10s long.
    i=0
    while i<len(trialStartTimes):
        if (trialEndTimes[i]-trialStartTimes[i]).total_seconds()<10:
            trialEndTimes.pop(i)
            trialStartTimes.pop(i)
            invDur.pop(i)
            windowDur.pop(i)
            i-=1
        i+=1
    
    if (len(trialStartTimes)!=16 or len(trialEndTimes)!=16):
        msg="The number of trials is not 16. There are %d trials."
        warn(msg)
    # Check that trials are all 30+/-1 seconds long.
    if not (np.abs(np.around([i.total_seconds() 
                              for i in np.diff( np.vstack(list(zip(trialStartTimes,trialEndTimes))),
                                  axis=1 ).ravel()])-30)<=1).all():
        warn("The trials are not all 30s long.")
    return trialStartTimes,trialEndTimes,invDur,windowDur
