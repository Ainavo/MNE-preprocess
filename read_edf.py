import mne
from mne.io import read_raw_edf
import numpy as np
import os

def edf_to_npy(subject_path):
    subeject_name = os.path.split(subject_path)[-1]
    left_fist_list = ['04', '08', '12'] #T1
    both_fists_list = ['06', '10', '14']   #or feet T2
    left_fist_data = []
    left_fist_labels = []
    for i in left_fist_list:
        edf_path = path+'\\'+subeject_name+'\\'+subeject_name+'R'+i+'.edf'
        raw = read_raw_edf(edf_path)
        event_id = {'T1':1, 'T2':2}
        events_from_annot, event_dict = mne.events_from_annotations(raw, event_id=event_id)
        epochs = mne.Epochs(raw, events_from_annot, tmin=0, tmax=4, baseline = (0., 0.))
        labels = epochs.events[:,-1]
        left_fist_data.append(epochs.get_data())
        left_fist_labels.append(labels)
    # left_fist_data = np.array(left_fist_data).reshape(left_fist_data[0].shape[0]+left_fist_data[1].shape[0]+left_fist_data[2].shape[0], 64, 641)
    # left_fist_labels = np.array(left_fist_labels).reshape(left_fist_labels[0].shape[0]+left_fist_labels[1].shape[0]+left_fist_labels[2].shape[0])
    left_fist_data1 = np.concatenate(left_fist_data, axis=0)
    left_fist_labels1 = np.concatenate(left_fist_labels, axis=0)
    print(left_fist_data1.shape)
    both_fists_data = []
    both_fists_labels = []
    for i in both_fists_list:
        edf_path = path+'\\'+subeject_name+'\\'+subeject_name+'R'+i+'.edf'
        raw = read_raw_edf(edf_path)
        event_id = {'T1':3, 'T2':4}
        events_from_annot, event_dict = mne.events_from_annotations(raw, event_id=event_id)
        epochs = mne.Epochs(raw, events_from_annot, tmin=0, tmax=4, baseline = (0., 0.))
        labels = epochs.events[:,-1]
        # print(labels)
        both_fists_data.append(epochs.get_data())
        both_fists_labels.append(labels)
    # both_fists_data = np.array(both_fists_data).reshape(both_fists_data[0].shape[0]+both_fists_data[1].shape[0]+both_fists_data[2].shape[0], 64, 641)
    # both_fists_labels = np.array(both_fists_labels).reshape(both_fists_labels[0].shape[0]+both_fists_labels[1].shape[0]+both_fists_labels[2].shape[0])
    both_fists_data1 = np.concatenate(both_fists_data, axis=0)
    both_fists_labels1 = np.concatenate(both_fists_labels, axis=0)
    data = np.concatenate((left_fist_data1, both_fists_data1), axis=0)
    label = np.concatenate((left_fist_labels1, both_fists_labels1), axis=0)
    npy_data_path = 'D:\DATA\Physionet_Database_npy'
    os.mkdir(npy_data_path+'\\'+subeject_name+'\\')
    data_path = npy_data_path+'\\'+subeject_name+'\\'+subeject_name+'R'+'_data.npy'
    label_path = npy_data_path+'\\'+subeject_name+'\\'+subeject_name+'R'+'_label.npy'
    np.save(data_path, data)
    np.save(label_path,label)

if __name__ =='__main__':
    data_path = os.walk('D:\DATA\Physionet_Database')
    for path, dir_list, file_list in (data_path):
        for dir_name in dir_list:
            subject_path = os.path.join(path, dir_name)
            # subeject_name = os.path.split(subject_path)[-1]
            edf_to_npy(subject_path)
            print(subject_path)
