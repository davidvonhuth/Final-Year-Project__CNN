#Importing libraries
import numpy as np 
import pandas as pd
import pyedflib
from pyedflib import highlevel as hl
import os
import pickle
import re
import time
import math
from keras.utils import to_categorical

start_time = time.time()

#File-reader function that will fetch the required EDF file and its labels
def readFile(file_nr):
    script_dir = os.path.dirname(__file__) 
    rel_path = "PSG_files/01-02-00" + file_nr + " PSG.edf"
    abs_file_path = os.path.join(script_dir, rel_path)

    psg_file = pyedflib.EdfReader(abs_file_path)
    ann_SS = pd.read_csv('SS_labels/01-02-00'  + file_nr +  ' SpindleE1.csv')
    edf_KC = pyedflib.EdfReader('KC_labels/01-02-00' + file_nr + ' KComplexesE1.edf')
    ann_KC = edf_KC.readAnnotations()

    return psg_file, ann_SS, ann_KC
    
#Function that fetches the indeces of the two requested EEG-channels (out of all available EEG channels)
def getChannelIndeces(psg_file):
    ch1 = 'EEG Fpz-CLE'
    ch2 = 'EEG Cz-CLE'
    all_labels = np.array(psg_file.getSignalLabels())
    idx_ch1 = int(np.where(all_labels == ch1)[0])
    idx_ch2 = int(np.where(all_labels == ch2)[0])
    return idx_ch1, idx_ch2


#Function that converts startpoint and duration values from the annotation data to values that correspond 
#with our desired segment size (e.g. 2 seconds) and overlapping amount (e.g. 1 second)
def convert(sps, durs, seg_size, overlap):
    new_sps = []
    new_durs = []
      
    for idx, sp in enumerate(sps):
        ds = sp/seg_size
        decimals = ds % 1

        if ds < 0.5:
            new_sps.append(ds)
            new_durs.append(durs[idx])
        else:
            if decimals > 0.5:
                mult1 = int(ds) * 2
                mult2 = int(ds) * 2 + 1
            else:
                mult1 = int(ds) * 2 - 1
                mult2 = int(ds) * 2

            sp1 = ds + (mult1 * overlap)  
            sp2 = ds + (mult2 * overlap)  
            new_sps.append(sp1)
            new_sps.append(sp2)
            new_durs.append(durs[idx])
            new_durs.append(durs[idx])

    return new_sps, new_durs


#Label-adding function that takes an (n * 3) numpy array (pre-filled with labels of the "Other" class, i.e. [1., 0., 0.]) 
#and fill in all (1 * 3) entries that are actually SSs and KCs, according to the provided annotation files
def add_label(label_array, startpoints, durations, size, th, label):
    both = np.array([0., 1., 1.])

    for idx, sp in enumerate(startpoints):
        if sp > len(label_array): 
            pass                      
        
        low_int = int(sp)
        up_int = low_int + 1
        duration = durations[idx]
        endpoint = sp + duration
        half_seg = low_int + 0.5


        if (label == (0., 0., 1.)).all(): #_____________________________________________________________________________K-Complex

            if endpoint < up_int:
                label_array[low_int] = label
                if sp < half_seg and (endpoint - half_seg) > (th * duration): 
                    label_array[up_int] = label
            else: 
                if (up_int - sp) > (th * duration):
                    label_array[low_int] = label
                if sp < half_seg and (endpoint - half_seg) > (th * duration): 
                    label_array[up_int] = label


        else: #_________________________________________________________________________________________________________Sleep Spindle

            if endpoint < (up_int):
                if (label_array[low_int] == (0., 0., 1.)).all() or (label_array[low_int] == (0., 1., 1.)).all(): label_array[low_int] = both
                else: label_array[low_int] = label

                if sp < half_seg and (endpoint - half_seg) > (th * duration):
                    if (label_array[up_int] == (0., 0., 1.)).all(): label_array[up_int] = both
                    else: label_array[up_int] = label

            else: 
                if (up_int - sp) > (th * duration): 
                    if (label_array[low_int] == (0., 0., 1.)).all() or (label_array[low_int] == (0., 1., 1.)).all() : label_array[low_int] = both
                    else: label_array[low_int] = label

                if sp < half_seg and (endpoint - half_seg) > (th * duration): 
                    if (label_array[up_int] == (0., 0., 1.)).all() or (label_array[low_int] == (0., 1., 1.)).all(): label_array[up_int] = both
                    else: label_array[up_int] = label

    return label_array


def prepare_data(psg_file, ann_SS, ann_KC, idx_ch1, idx_ch2):

    #Initializing variables
    label_dict = {'ss': np.array([0., 1., 0.]), 'kc': np.array([0., 0., 1.]), 'other': np.array([[1., 0., 0.]])}
    seg_info = {'size' : 2, 'th' : 0.5, 'overlap_label': 0.5, 'overlap_signal': 256}
    sampling_rate = 256 * seg_info['size']

    #Reading the two chosen channels from the PSG-file (Polysomnography-file)
    sig_ch1 = psg_file.readSignal(idx_ch1)
    sig_ch2 = psg_file.readSignal(idx_ch2) 
    signal = sig_ch1 - sig_ch2 #Creating a derivation signal between the two chosen channels

    #Overlapping --> Creating new segments by overlapping existing ones
    segs_before_ol = int(len(signal)/sampling_rate)
    meaningful_signals_idx = sampling_rate * segs_before_ol
    signal = np.copy(signal[:meaningful_signals_idx])
    total_segments = int((len(signal) - sampling_rate) / seg_info['overlap_signal'] + 1) 
    ol_signal = np.copy(np.lib.stride_tricks.as_strided(signal, strides=(seg_info['overlap_signal']*8, 8), shape=(total_segments,sampling_rate)))
    
    #Converting startpoints and durations from original segments to that of overlapping & larger segments
    SS_startpoints, SS_durations = convert(list(ann_SS['start']), list(ann_SS['duration']), seg_info['size'], seg_info['overlap_label'])
    KC_startpoints, KC_durations = convert(list(ann_KC[0]), list(ann_KC[1]), seg_info['size'], seg_info['overlap_label']) 

    #Labelling
    label_array = label_dict['other'].repeat(total_segments, axis=0) 
    label_array = add_label(label_array, KC_startpoints, KC_durations, seg_info['size'], seg_info['th'], label_dict['kc'])
    label_array = add_label(label_array, SS_startpoints, SS_durations, seg_info['size'], seg_info['th'], label_dict['ss'])

    #Printing information
    print('\nTotal rounded 2-second segments BEFORE overlapping: ',segs_before_ol)
    print('Total rounded 2-second segments AFTER overlapping: ', ol_signal.shape)
    print('Annotations array shape: ', label_array.shape)
    print('Number of added labels per column --> (other, ss, kc): ', label_array.sum(axis=0))
    print('Number of actual SSs: ', len(SS_startpoints))
    print('Number of actual KCs: ', len(KC_startpoints))
    print('-------------------------------------------------------------------------------------------------------------')
    
    return ol_signal, label_array


#Serializing EEG data and labels into a pickled file
def pickle_objects(all_signals, all_labels):
    fname = 'pre_2-sec_0.5_EEG_data.pickle'

    with open(fname, 'wb') as f:
        pickle.dump(all_signals, f)
        pickle.dump(all_labels, f)

    file = open(fname, 'rb')
    signals = pickle.load(file)
    labels = pickle.load(file)

    print('_____________________________________________')
    print('Pickled signals: ', signals.shape)
    print('Pickled labels: ',labels.shape)


  


# **--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**
#-------------------------------------------< MAIN FUNCTION >------------------------------------------#
def main():
    nr_subjects = 19

    for i in range(nr_subjects):
        if i < 9: 
            subject_nr = '0' + str(i+1)
        else: 
            subject_nr = str(i+1)

        print('\nAnnotating PSG file number-----------------------------------------------------> ' + subject_nr)

        psg_file, ann_SS, ann_KC = readFile(subject_nr)
        idx_ch1, idx_ch2 = getChannelIndeces(psg_file)
        signal, label_array = prepare_data(psg_file, ann_SS, ann_KC, idx_ch1, idx_ch2)

        if i == 0:
            all_signals = signal
            all_labels = label_array
        else:
            all_signals = np.append(all_signals, signal, 0)
            all_labels = np.append(all_labels, label_array, 0)
        print('\n')

        print('Nr. of total others: ', sum((all_labels == (1., 0., 0.)).all(axis=1)))
        print('Nr. of total ss: ', sum((all_labels == (0., 1., 0.)).all(axis=1)))
        print('Nr. of total kc: ', sum((all_labels == (0., 0., 1.)).all(axis=1)))
        print('Nr. of total both: ', sum((all_labels == (0., 1., 1.)).all(axis=1)))
    pickle_objects(all_signals, all_labels)

    print("\n------------ %s seconds -----------" % np.round((time.time() - start_time),3))
# **--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**


#Calling main function
# ================================
if __name__ == '__main__':
    main()
# ================================
