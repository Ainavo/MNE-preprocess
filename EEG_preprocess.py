import numpy as np
import mne 
from mne.preprocessing import ICA 
from mne.time_frequency import tfr_morlet
import scipy.io as scio

#input_shape=(channels, samples)
def eeg_preprocess(data_model='YS', data_path=None):
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
                #课题室数据集
                structured_eeg = scio.loadmat(data_path)
                eeg_data = structured_eeg['EEG']['data'].item()
                raw = mne.io.RawArray(data=eeg_data[0:59], info=info)
                raw.plot(n_channels=10, 
                        scalings=50,
                        duration=3,
                        title='Data from arrays',
                        show=True, 
                        block=True,
                        )
                # raw.plot_psd(fmax=60, average=True)  #平均功率谱密度(fmax:频率)
                # raw.plot_sensors(ch_type='eeg', show_names=True)  #源电极
                # raw.plot_psd_topo(fmax=60) #电极psd
                raw = raw.filter(l_freq=0.1, h_freq=30, method='fir')   #method='iir
                # raw.plot_psd(fmax=60, average=True)

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
                # ica.plot_overlay(raw_for_ica, exclude=reject)  #ica前后区别图
                ica.apply(raw)


        #08公开数据
        if data_model =='08BCI':
                ch_names = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz',
                        ]
                ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                        'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                        'eeg','eeg',
                ]

                sfreq = 250
                info = mne.create_info(
                        ch_names=ch_names,
                        ch_types=ch_types,
                        sfreq=sfreq
                )
                info.set_montage('standard_1020')
                structured_data = scio.loadmat(data_path)
                eeg_data = structured_data['data'][0, 4]['X'].item().transpose((1, 0))
                raw = mne.io.RawArray(data=eeg_data[0:22], info=info)
                raw.plot(n_channels=10, 
                        scalings=50,
                        duration=3,
                        title='Data from arrays',
                        show=True, 
                        block=True,
                        )
                # raw.plot_psd(fmax=60, average=True)  #平均功率谱密度(fmax:频率)
                # raw.plot_sensors(ch_type='eeg', show_names=True)  #源电极
                # raw.plot_psd_topo(fmax=60) #电极psd
                raw = raw.filter(l_freq=0.1, h_freq=30, method='fir')   #method='iir
                # raw.plot_psd(fmax=60, average=True)

                #插值坏导
                raw = raw.interpolate_bads()
                ica = ICA(n_components=15, max_iter='auto')
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
                # ica.plot_overlay(raw_for_ica, exclude=reject)  #ica前后区别图
                ica.apply(raw)

if __name__ == '__main__':
        ys_data_path = r'D:\DATA\脑电数据\5.17脑电数据\郑力文\EEG.mat'
        # eeg_preprocess(data_model='YS', data_path=ys_data_path)
        bci_data_path = r'D:\DATA\BCI Competition IV dataset 2a\A01T.mat'
        eeg_preprocess(data_model='08BCI', data_path=bci_data_path) #08BCI以A01T 05为例