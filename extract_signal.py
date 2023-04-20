"""
Extract signals
This script contains functions for
1) Extracting spikes or signals from raw signal
    - MUA()
    - MUAe()
    - LFP()
    - nLFP()
2) Loading raw signal
    - RAW()

Authors: Matěj Voldřich

Most of these functions are modified functions from
https://gin.g-node.org/NIN/V1_V4_1024_electrode_resting_state_data/src/master/code/python_scripts/signal_processing
and only used for figures.
"""
from elephant.signal_processing import butter
from neo import Segment, AnalogSignal, Block
import quantities as pq
import elephant
import numpy as np
from neo import NixIO, BlackrockIO


def LFP(path):
    if "ns6" in path:
        bl = BlackrockIO(path).read_block()
    elif "nix" in path:
        bl = NixIO(path, "ro").read_block()
    seg = bl.segments[0]

    ansignals = []
    for i in range(seg.analogsignals[0].shape[1]):
        anasig = seg.analogsignals[0][:, i]
        anasig = anasig.reshape(anasig.shape[0])
        # 2. Filter the signal between 1Hz and 150Hz
        anasig = butter(anasig, lowpass_freq=150.0 * pq.Hz)

        # 3. Downsample signal from 30kHz to 500Hz resolution (factor 60)
        anasig = anasig.downsample(60, ftype='fir')

        # 4. Bandstop filter the 50, 100 and 150 Hz frequencies
        # Compensates for artifacts from the European electric grid
        for fq in [50, 100, 150]:
            anasig = butter(anasig, highpass_freq=(fq + 2) * pq.Hz, lowpass_freq=(fq - 2) * pq.Hz)
        # seg.analogsignals[0][:, i] = anasig
        ansignals.append(anasig)
    ansignals = np.array(ansignals)
    ansignals = ansignals.reshape((ansignals.shape[0], ansignals.shape[1])).T
    seg = Segment()
    seg.analogsignals.append(AnalogSignal(np.array(ansignals), units=pq.mV, sampling_rate=500*pq.Hz))
    bl = Block()
    bl.segments.append(seg)
    return bl


def MUAe(path):
    if "ns6" in path:
        bl = BlackrockIO(path).read_block()
    elif "nix" in path:
        bl = NixIO(path, "ro").read_block()
    seg = bl.segments[0]
    ansignals = []
    for i in range(seg.analogsignals[0].shape[1]):
        anasig = seg.analogsignals[0][:, i]

        # 2. Filter the signal between 500Hz and 9000Hz
        anasig = butter(anasig,
                        highpass_freq=500.0 * pq.Hz,
                        lowpass_freq=9000.0 * pq.Hz,
                        fs=anasig.sampling_rate.magnitude)
        # 3. Rectify the filtered wave
        anasig = anasig.rectify()

        # 4. Low pass filter at 200Hz
        anasig = butter(anasig,
                        lowpass_freq=200.0 * pq.Hz,
                        fs=anasig.sampling_rate.magnitude)

        # 5. Downsample signal from 30kHz to 1kHz resolution (factor 30)
        anasig = anasig.downsample(30, ftype='fir')

        # 6. Bandstop filter the 50, 100 and 150 Hz frequencies
        # Compensates for artifacts from the European electric grid
        for fq in [50, 100, 150]:
            anasig = butter(anasig,
                            highpass_freq=(fq + 2) * pq.Hz,
                            lowpass_freq=(fq - 2) * pq.Hz,
                            fs=anasig.sampling_rate.magnitude)
        ansignals.append(anasig)
    ansignals = np.array(ansignals)
    ansignals = ansignals.reshape((ansignals.shape[0], ansignals.shape[1])).T
    seg = Segment()
    seg.analogsignals.append(AnalogSignal(np.array(ansignals), units=pq.mV, sampling_rate=1000 * pq.Hz))
    bl = Block()
    bl.segments.append(seg)
    return bl


def nLFP(path, thr_factor):
    bl = LFP(path)
    seg = bl.segments[0]

    before_art_removal = []
    trials = [[seg.analogsignals[0].t_start,
               seg.analogsignals[0].t_stop]]
    for anasig in seg.analogsignals[0].T:
        for tnr, trial in enumerate(trials):
            if tnr == 0:  # define the threshold based on the spontaneous activity
                thr = thr_factor * np.std(anasig)
                if thr_factor > 0:
                    sign_ext = 'above'
                else:
                    sign_ext = 'below'

            # extract peaks
            anasig = AnalogSignal(np.array(anasig), units=pq.mV, sampling_rate=500*pq.Hz)
            st = elephant.spike_train_generation.peak_detection(
                anasig,
                threshold=np.array(thr) * anasig.units,
                sign=sign_ext)
            bl.segments[tnr].spiketrains.append(st)
            before_art_removal.append(len(st))
    return bl


def MUA(path, thr_factor):
    if "ns6" in path:
        bl = BlackrockIO(path).read_block()
    elif "nix" in path:
        bl = NixIO(path, "ro").read_block()
    seg = bl.segments[0]

    ansig = seg.analogsignals[0]
    for i in range(ansig.shape[1]):
        anasig = ansig[:, i]
        # band-pass filter
        anasig_filt = elephant.signal_processing.butter(
            anasig, fs=30000,
            highpass_freq=300.,
            lowpass_freq=6000.)
        ansig[:, i] = anasig_filt
        thr = thr_factor * np.std(anasig_filt)
        if thr_factor > 0:
            sign_ext = 'above'
        else:
            sign_ext = 'below'

        # extract peaks
        anasig_filt = AnalogSignal(np.array(anasig), units=pq.mV, sampling_rate=30000 * pq.Hz)
        st = elephant.spike_train_generation.peak_detection(
            anasig_filt,
            threshold=np.array(thr) * anasig_filt.units,
            sign=sign_ext)
        seg.spiketrains.append(st)
    return bl


def RAW(path):
    if "ns6" in path:
        bl = BlackrockIO(path).read_block()
    elif "nix" in path:
        bl = NixIO(path, "ro").read_block()
    return bl
