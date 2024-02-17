import numpy as np
import mne 
from mne.preprocessing import ICA 
from mne.time_frequency import tfr_morlet
import scipy.io as scio
import pickle

#input_shape=(channels, samples)
def eeg_preprocess(data_model='3', data_path=None):
        if data_path==None:
                print('请输入数据的路径')
        #博睿康59导
        if data_model == 'YS':
                ch_names = ['Fpz','Fp1','Fp2','AF3','AF4','AF7','AF8','Fz','F1','F2','F3','F4','F5','F6','F7','F8','FCz','FC1','FC2','FC3','FC4','FC5','FC6','FT7','FT8','Cz','C1','C2','C3','C4','C5','C6','T7','T8','CP1','CP2','CP3','CP4','CP5','CP6','TP7','TP8','Pz','P3','P4','P5','P6','P7','P8','POz',
                            'PO3','PO4','PO5','PO6','PO7','PO8','Oz','O1','O2']
                ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                            'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                            'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                            'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                            'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                            'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg'
                ]

                sfreq = 1000
                info = mne.create_info(
                        ch_names=ch_names,
                        ch_types=ch_types,
                        sfreq=sfreq)
                info.set_montage('standard_1020')
                trigger_list = [250, 251, 240, 241, 243, 243, 201, 202, 203]

                dp = open(data_path, 'rb')
                data = pickle.load(dp)
                trigger_id_list = []
                trigger_idx_list = []
                for trigger_idx, trigger_id in enumerate(data[-1]):
                        # if trigger_id != 0:
                        #     print(trigger_id)
                        if trigger_id in trigger_list:
                                trigger_idx_list.append(trigger_idx)
                                trigger_id_list.append(trigger_id)
                print(len(trigger_id_list), len(trigger_idx_list))
                events = np.zeros((len(trigger_id_list), 3), int)
                events[:, 0] = np.squeeze(trigger_idx_list)
                events[:, 2] = np.squeeze(trigger_id_list)
                #课题室数据集
                # structured_eeg = scio.loadmat(data_path)
                # eeg_data = structured_eeg['EEG']['data'].item()
                # events_list = structured_eeg["EEG"]["events"]
                # events = np.zeros((len(events_list), 3), int)
                # for i in range(len(events_list)):
                #         events[i, 0] = events_list[i, 0][1]
                #         events[i, 2] = events_list[i, 0][0]
                raw = mne.io.RawArray(data=data[:59], info=info)
                raw.plot(n_channels=59, 
                        scalings=50,
                        title='YSU EEG DATA',
                        show=True, 
                        block=True,
                        )
                raw.compute_psd(fmax=60).plot()  #平均功率谱密度(fmax:频率)
                raw.plot_sensors(ch_type='eeg', show_names=True)  #源电极
                raw.plot_psd_topo(fmax=60) #电极psd
                raw = raw.filter(l_freq=0.1, h_freq=30, method='fir')   #method='iir
                raw.compute_psd(fmax=60).plot()

                #插值坏导
                raw = raw.interpolate_bads()
                ica = ICA(n_components=20, max_iter='auto')
                raw_for_ica = raw.copy().filter(l_freq=1, h_freq=None)
                ica.fit(raw_for_ica)
                ica.plot_sources(raw_for_ica)
                ica.plot_components(inst=raw)
                x = input('输入拒绝的成分：')
                if len(x) == 0:
                        reject=[]
                elif len(x) == 1:
                        reject = [int(x)]
                else:
                        xlist=x.split(',')
                        reject = [int(xlist[i]) for i in range(len(xlist))]
                ica.exclude = reject
                ica.plot_overlay(raw_for_ica, exclude=reject)  #ica前后区别图
                ica.apply(raw)
                event_id = dict(lefthand=201, righthand=202, bothfeet=203)
                picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, ecg=False,
                        exclude='bads')
                epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4, proj=False, baseline=(None, None),
                                    picks=picks, preload=False, detrend=None)
                epochs.plot(n_epochs=3,
                        events=True, 
                        event_id=True, 
                        scalings=50,
                        title='YSU EEG DATA',
                        show=True, 
                        block=True,
                        butterfly=True,)
                epochs.compute_psd().plot_topomap()


        # #08公开数据
        if data_model =='2':
                ch_names = ['Fpz','Fp1','Fp2','AF3','AF4','AF7','AF8','Fz','F1','F2','F3','F4','F5','F6','F7','F8','FCz','FC1','FC2','FC3','FC4','FC5','FC6','FT7','FT8','Cz','C1','C2','C3','C4','C5','C6','T7','T8','CP1','CP2','CP3','CP4','CP5','CP6','TP7','TP8','Pz','P3','P4','P5','P6','P7','P8','POz',
                                'PO3','PO4','PO5','PO6','PO7','PO8','Oz','O1','O2']
                ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                        'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                        'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                        'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                        'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                        'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg'
                ]

                sfreq = 1000
                info = mne.create_info(
                        ch_names=ch_names,
                        ch_types=ch_types,
                        sfreq=sfreq)
                info.set_montage('standard_1020')
                #课题室数据集
                structured_eeg = scio.loadmat(data_path)
                eeg_data = structured_eeg['EEG']['data'].item()
                events_list = structured_eeg["EEG"]["event"][0][0]
                events = np.zeros((len(events_list), 3), int)
                # print(events_list)
                for i in range(len(events_list)):
                        events[i, 0] = events_list[i, 0][1]
                        events[i, 2] = events_list[i, 0][0]
                # print(events)
                raw = mne.io.RawArray(data=eeg_data[0:59], info=info)
                raw.plot(n_channels=59, 
                        scalings=25,
                        title='YSU EEG Data',
                        show=True, 
                        block=True,
                        )
                raw.plot_psd(fmax=80, average=False)  #平均功率谱密度(fmax:频率)
                raw.plot_sensors(ch_type='eeg', show_names=True)  #源电极
                raw.plot_psd_topo(fmax=200) #电极psd
                raw = raw.filter(l_freq=0.1, h_freq=30, method='fir')   #method='iir
                raw.plot_psd(fmax=80, average=False)

                # #插值坏导
                raw = raw.interpolate_bads()
                # ica = ICA(n_components=20, max_iter='auto')
                # raw_for_ica = raw.copy().filter(l_freq=1, h_freq=None)
                # ica.fit(raw_for_ica)
                # ica.plot_sources(raw_for_ica)
                # ica.plot_components(inst=raw)
                # x = input('输入拒绝的成分：')
                # if len(x) == 0:
                #         reject=[]
                # elif len(x) == 1:
                #         reject = [int(x)]
                # else:
                #         xlist=x.split(',')
                #         reject = [int(xlist[i]) for i in range(len(xlist))]
                # ica.exclude = reject
                # ica.plot_overlay(raw_for_ica, exclude=reject)  #ica前后区别图
                # ica.apply(raw)
                event_id = dict(lefthand=1, righthand=2)
                picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, ecg=False,
                        exclude='bads')
                epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4, proj=False, baseline=(0, 0.8),
                                    picks=picks, preload=True, detrend=1)
                ica = ICA(n_components=20, max_iter='auto')
                epochs_for_ica = epochs.copy().filter(l_freq=1, h_freq=None)
                ica.fit(epochs_for_ica)
                ica.plot_sources(epochs_for_ica)
                ica.plot_components(inst=epochs, title='ICA components')
                x = input('输入拒绝的成分：')
                if len(x) == 0:
                        reject=[]
                elif len(x) == 1:
                        reject = [int(x)]
                else:
                        xlist=x.split(',')
                        reject = [int(xlist[i]) for i in range(len(xlist))]
                ica.exclude = reject
                ica.plot_overlay(raw, exclude=reject)  #ica前后区别图
                ica.apply(epochs)

if __name__ == '__main__':
        ys_data_path = r'F:\S08.pkl'
        # eeg_preprocess(data_model='YS', data_path=ys_data_path)
        # bci_data_path = r'F:\EEG.mat'
        bci_data_path = r"H:\参考数据\5.17脑电数据\张孜涵\EEG.mat"
        eeg_preprocess(data_model='2', data_path=bci_data_path) #08BCI以A01T 05为例