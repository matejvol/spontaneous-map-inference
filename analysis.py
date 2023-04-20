"""
Analysis
This script contains functions for
1) generating frames
    - get_frames()
    - get_frames_MUAe()
    - get_frames_lfp()
    - get_frames_nLFP()
2) data interpolation
    - interpolate_data()
3) generating correlation maps
    - calculate_correlation_maps_pool()
4) generating spontaneous maps
    - PCA_map()

Authors: Karolína Korvasová, Matěj Voldřich
"""
import os
import neo
import pandas as pd
import numpy as np
import scipy
import pickle
import quantities as pq
import helper


def get_frames(path, params, tag=''):
    """
    Generate frames (activation patterns) from spikes.

    :param path: path to MUA or SUA spikes path
    :param params: analysis parameters
    :param tag: "MUA" or "SUA"
    :return: frames, rates, spike_counts, duration, ts
    """
    if 'MUA' in tag or 'SUA' in tag:
        bl = helper.load_spikes(path, tag)
    else:
        print('Unrecognized data format')
        exit()

    seg_orig = bl.segments[0]

    if params['event_binsize'] == 'median':
        split_const = params['split_constant']
    else:
        split_const = 2 * params['event_binsize']

    ts = []  # for events=False the times are returned, should be always returned

    if params['min_eyes_closed'] >= -0.001:
        seg = neo.Segment()

        if '090817' in path:
            metadata_path = '{}/V1_V4_1024_electrode_resting_state_data/metadata/epochs/epochs_L_RS_090817.csv'.format(
                params['data_folder_monkey'])
        elif '100817' in path:
            metadata_path = '{}/V1_V4_1024_electrode_resting_state_data/metadata/epochs/epochs_L_RS_100817.csv'.format(
                params['data_folder_monkey'])
        elif '250717' in path:
            metadata_path = '{}/V1_V4_1024_electrode_resting_state_data/metadata/epochs/epochs_L_RS_250717.csv'.format(
                params['data_folder_monkey'])
        elif '140819' in path:
            metadata_path = '{}/V1_V4_1024_electrode_resting_state_data/metadata/epochs/epochs_A_RS_140819.csv'.format(
                params['data_folder_monkey'])
        elif '150819' in path:
            metadata_path = '{}/V1_V4_1024_electrode_resting_state_data/metadata/epochs/epochs_A_RS_150819.csv'.format(
                params['data_folder_monkey'])
        elif '160819' in path:
            metadata_path = '{}/V1_V4_1024_electrode_resting_state_data/metadata/epochs/epochs_A_RS_160819.csv'.format(
                params['data_folder_monkey'])

        else:
            print('Unknown metadata path for epochs with closed eyes.')
            exit()
        metadata = pd.read_csv(metadata_path)

        print('Extracting eyes closed.')
        ids_closed = np.where(metadata['state'] == 'Closed_eyes')[0]

        duration = 0
        for ix in ids_closed:
            timeint = metadata['t_stop'][ix] - metadata['t_start'][ix]
            if timeint > params['min_eyes_closed']:  # closed eyes for more than x
                duration += timeint

        for stnr, st in enumerate(seg_orig.spiketrains):
            newspikes = []
            subtract = 0.
            for ixnr, ix in enumerate(ids_closed):
                st_eyes_closed = st[(st > metadata['t_start'][ix]) &
                                    (st < metadata['t_stop'][ix])].magnitude
                if ixnr == 0:
                    subtract += metadata['t_start'][ix]
                else:
                    subtract += metadata['t_start'][ix] - metadata['t_stop'][ids_closed[ixnr - 1]]

                timeint = metadata['t_stop'][ix] - metadata['t_start'][ix]
                if timeint > params['min_eyes_closed']:
                    if len(st_eyes_closed) > 0:
                        # newspikes.extend((st_eyes_closed).tolist()) #-subtract).tolist())
                        newspikes.extend((st_eyes_closed - subtract + split_const).tolist())
                        duration += split_const
                else:
                    subtract += timeint
            tstart = 0.
            # tstop = seg_orig.spiketrains[0].t_stop.magnitude # duration
            tstop = duration
            seg.spiketrains.append(neo.SpikeTrain(np.array(newspikes) * pq.s,
                                                  t_start=tstart * pq.s,
                                                  t_stop=tstop * pq.s))
        print('Extracted {:.2f} min with closed eyes.'.format(duration / 60.))
    else:
        # copy spikes
        seg = seg_orig
        tstart = seg_orig.spiketrains[0].t_start.magnitude
        tstop = seg_orig.spiketrains[0].t_stop.magnitude
        duration = tstop - tstart

    rates = []
    spike_counts = []

    for ch, st in enumerate(seg.spiketrains):
        rates.append(len(st) / duration)
        spike_counts.append(len(st))
        spike_counts.append(len(st))

    if 'filter' in params['frame_type']:

        # import elephant
        # t_res = params['event_binsize']  # ms

        bins = np.arange(seg.t_start, seg.t_stop, params['event_binsize'] * pq.s)
        binned_spiketrains = []
        for st in seg.spiketrains:
            hist, bins = np.histogram(st, bins=bins)
            binned_spiketrains.append(hist)

        filtered_traces = []
        active = []
        for bst in binned_spiketrains:
            filtered_trace = helper.convolve_signal(bst, bins[:-1], params, tau=params['imaging_kernel'])
            filtered_traces.append(filtered_trace)
            active_thr = np.mean(filtered_trace) + 1. * np.std(filtered_trace)
            active.append(filtered_trace > active_thr)

        filtered_traces = np.squeeze(np.array(filtered_traces))
        active = np.array(active)

        zeros = np.where(np.sum(active, axis=0) < params['active_frames_thr'] * active.shape[0])[0]
        print('Typical event duration: 50ms *', np.mean(np.diff(zeros)))

        # active pixel 4xstd above its mean value
        # 80% pixels active for event detection

        bs = params['event_binsize']

        frames = []
        for ix in range(len(zeros) - 1):
            # if zeros[ix + 1] - zeros[ix] > 1:
            if (((zeros[ix + 1] - zeros[ix]) >= params['min_event_duration'] / bs) and
                    ((zeros[ix + 1] - zeros[ix]) <= params['max_event_duration'] / bs)):
                if params['frame_type'] == 'filter_max':
                    frames.append(np.max(filtered_traces[:, zeros[ix]:zeros[ix + 1]], axis=1))
                elif params['frame_type'] == 'filter_sum':
                    frames.append(np.sum(filtered_traces[:, zeros[ix]:zeros[ix + 1]], axis=1))
                else:
                    print('Frame type not known.')

        frames = np.array(frames)

    else:
        # first extract events as Smith et al. (i.e. avalanches)
        allspikes = []
        for st in seg.spiketrains:
            # pool spikes
            allspikes.extend(st.magnitude.tolist())

        allspikes = np.array(allspikes)

        if len(allspikes) == 0:
            return [], [], [], [], []
        if params['event_binsize'] == 'median':
            dt = np.median(np.diff(np.sort(allspikes)))  # 1000.*
            print('Bin size: {} * {}'.format(dt, st.units))
        else:
            dt = params['event_binsize']  # 0.01 #np.mean(np.diff(np.sort(allspikes)))
            print(f'Bin size: {dt} s.')

        bins = np.arange(tstart, tstop, dt)
        hist, bins = np.histogram(allspikes, bins)
        print('NUmber of spikes:', len(allspikes))

        print('Shape of the histogram:', hist.shape)
        if params['events']:

            bs = bins[1] - bins[0]

            # find zeros in the histogram
            zeros_all = np.where(hist == 0)[0]
            zeros = zeros_all
            event_ints = []

            frames = []
            events_wrong = []
            for ix in range(len(zeros) - 1):
                event_ints.append([bins[zeros[ix]], bins[zeros[ix + 1]]])
                events_wrong.append([zeros[ix], zeros[ix + 1]])

                if params['frame_type'] == 'max':
                    histev = hist[zeros[ix]:zeros[ix + 1]]
                    binsev = bins[zeros[ix]:zeros[ix + 1]]  # left edges of the bins
                    active_frame_ix = np.argmax(histev)
                    active_int = [binsev[active_frame_ix], binsev[active_frame_ix] + bs]
                    frame = []
                    for st in seg.spiketrains:
                        stfr = st[(st > active_int[0]) & (st < active_int[1])]
                        frame.append(len(stfr))
                elif params['frame_type'] == 'sum':
                    frame = []
                    # frame_wrong = []
                    for st in seg.spiketrains:
                        stfr = st[(st > bins[zeros[ix]]) & (st < bins[zeros[ix + 1]])]
                        # stfr = st[(st > zeros[ix]) & (st < zeros[ix + 1])] # wrong
                        # stfr = st[(st > bins[zeros[ix][0]]) & (st < bins[zeros[ix][1]])]
                        frame.append(len(stfr))

                        # stfr = st[(st > zeros[ix]) & (st < zeros[ix + 1])]
                        # stfr = st[(st > zeros[ix][0]) & (st < zeros[ix][1])]
                        # frame_wrong.append(len(stfr))
                else:
                    print('Frame type not known.')

                if np.sum(frame) > 0:
                    frames.append(frame)
                    ts.append((zeros[ix] + zeros[ix + 1]) / 2.)

            frames = np.array(frames)
            ts = np.array(ts)
        else:
            frames = []
            for st in seg.spiketrains:
                bins = np.arange(tstart, tstop, dt)
                hist_chn, bins = np.histogram(st, bins)
                frames.append(hist_chn)
            frames = np.array(frames).T
            ts = bins[:-1] + dt / 2.

    counts = np.max(frames, axis=1)
    print('Maximum frame:', np.max(counts))
    print('Active frames have shape', frames.shape)
    print('Times have shape:', ts.shape)
    print("Duration:", duration)
    return frames, rates, spike_counts, duration, ts


def get_frames_MUAe(paths: list, params):
    """
    Generate frames (activation patterns) for MUAe signal.

    :param paths: list MUAe signal paths
    :param params: analysis parameters
    :return: frames
    """
    # merge analog signals from all recordings
    from elephant.signal_processing import butter
    fs = 1000
    analog_sigs = []
    for path in paths:
        sig = helper.extract_signal_eyes_closed(path, params, fs)
        if params['subtract_mean']:
            sig = (sig.T - np.mean(sig, axis=1)).T
        muae_f = params['muae_filter']
        if muae_f > 0:
            sig = butter(sig, lowpass_freq=int(muae_f) * pq.Hz, fs=1000 * pq.Hz)
            sig = neo.AnalogSignal(sig, sampling_rate=1000 * pq.Hz, units=pq.uV).downsample(int(500 / muae_f),
                                                                                            ftype='fir')
        else:
            sig = neo.AnalogSignal(sig, sampling_rate=1000 * pq.Hz, units=pq.uV).downsample(2, ftype='fir')
        analog_sigs.append(sig)
    frames = np.array(analog_sigs[0])
    for i in range(1, len(analog_sigs)):
        frames = np.concatenate((frames, analog_sigs[i]))
    return np.array(frames)


def get_frames_lfp(paths, params):
    """
    Generate frames (activation patterns) for LFP signal.

    :param paths: list LFP signal paths
    :param params: analysis parameters
    :return: frames
    """
    from scipy.signal import welch
    analog_sigs = []
    fs = 500
    for path in paths:
        sigs = helper.extract_signal_eyes_closed(path, params, fs)
        if params['subtract_mean']:
            sigs = (sigs.T - np.mean(sigs, axis=1)).T
        analog_sigs.append(sigs)
    sigs_combined = np.array(analog_sigs[0])
    for i in range(1, len(analog_sigs)):
        sigs_combined = np.concatenate((sigs_combined, analog_sigs[i]))
    step = fs * params['event_binsize']
    frames = np.zeros((int(sigs_combined.shape[0] / step), 64))
    for i in range(frames.shape[0]):
        for j in range(frames.shape[1]):
            frames[i, j] = np.sum(welch(sigs_combined[int(i * step):int((i + 1) * step), j], fs=fs)[1])
    return frames


def get_frames_nLFP(paths, params):
    """
    Extract nLFP spikes from LFP signal and generate frames (activation patterns).

    :param paths: list LFP signal paths
    :param params: analysis parameters
    :return: frames
    """
    analog_sigs = []
    fs = 500
    # load and join recordings
    for path in paths:
        sigs = helper.extract_signal_eyes_closed(path, params, fs)
        if params['subtract_mean']:
            sigs = (sigs.T - np.mean(sigs, axis=1)).T
        analog_sigs.append(sigs)
    sigs_combined = np.array(analog_sigs[0])
    for i in range(1, len(analog_sigs)):
        sigs_combined = np.concatenate((sigs_combined, analog_sigs[i]))
    sigs_combined = neo.AnalogSignal(sigs_combined, units=pq.uV, sampling_rate=fs * pq.Hz)

    # extract nLFP spikes
    sts = helper.calculate_nLFP(sigs_combined, params['thr_factor']).segments[0].spiketrains

    # calculate frames
    step = fs * params['event_binsize']
    frames = np.zeros((int(sigs_combined.shape[0] / step), 64))
    for i in range(frames.shape[0]):
        t1 = i * params['event_binsize']
        t2 = (i + 1) * params['event_binsize']
        for j in range(frames.shape[1]):
            frames[i, j] = np.sum(np.logical_and(sts[j] >= t1, sts[j] < t2))
    return frames


def interpolate_data(content, params, tag=''):
    """
    Interpolate fake channels for evoked orientation map or correlation maps.

    :param content: 2D data array with proper spatial organization
    :param params: analysis parameters
    :param tag: 'orientation' for spontaneous/orientation map or nothing for correlation maps
    :return: 1D interpolated data, all indices, real channel indices, electrode indices
    """
    average = np.mean
    if "orientation" in tag:
        average = lambda angle: scipy.stats.circmean(angle, high=180)
    If = params["interpolation_factor"]
    n = content.shape[0]

    def get_corners(row, col, lim):
        """
        Get valid corners or -1 if in untunned channel
        """
        check_pos = lambda x: 0 <= x < lim
        corners = []
        if row % 2 == 0 and col % 2 == 0:
            # interpolate corners
            if col % 2 == 0:
                for i in range(-1, 2, 2):
                    for j in range(-1, 2, 2):
                        if check_pos(row + i) and check_pos(col + j):
                            corners.append((row + i, col + j))
                    # interpolate vertically
        elif row % 2 == 0 or col % 2 == 0:
            for i in range(-1, 2, 2):
                if check_pos(row + i):
                    corners.append((row + i, col))
            for j in range(-1, 2, 2):
                if check_pos(col + j):
                    corners.append((row, col + j))
        if row % 2 == 1:  # to keep information from previous interpolation step at the start of the list
            corners.reverse()
        return corners

    new_dim = 2 ** If * n + 2 ** If - 1
    inter_content = np.zeros(shape=(new_dim, new_dim))

    # initialize real values
    # works only for the monkey
    offset = int((new_dim - n) / (n + 1))
    real_channel_indices = []
    real_electrode_indices = np.zeros(64)

    for a in range(1, n + 1):
        for b in range(1, n + 1):
            row = a * (offset + 1) - 1
            col = b * (offset + 1) - 1
            inter_content[row, col] = content[a - 1, b - 1]
            real_channel_indices.append((row, col))

    for _ in range(1, If + 1):
        n = 2 * n + 1
        offset = int((new_dim - n) / (n + 1))
        for r in range(int((n + 1) / 2)):
            for c in range(int((n + 1) / 2)):
                # interpolate value - stage one (only even coords)
                row, col = 2 * r, 2 * c
                corners = get_corners(row, col, n)
                values = [inter_content[int((a + 1) * (offset + 1) - 1), int((b + 1) * (offset + 1) - 1)]
                          for a, b in corners]
                values = [val for val in values if val != -1]
                inter_content[(row + 1) * (offset + 1) - 1, (col + 1) * (offset + 1) - 1] = -1 if len(values) == 0 \
                    else average(values)

        # stage II: [even, odd] - vertical
        for r in range(int((n + 1) / 2)):
            for c in range(int((n - 1) / 2)):
                row, col = 2 * r, (2 * c) + 1
                corners = get_corners(row, col, n)
                values = [inter_content[int((a + 1) * (offset + 1) - 1), int((b + 1) * (offset + 1) - 1)]
                          for a, b in corners]
                values = [val for val in values if val != -1]
                inter_content[(row + 1) * (offset + 1) - 1, (col + 1) * (offset + 1) - 1] = -1 if len(values) == 0 \
                    else average(values)

        # stage II: [odd, even] - horizontal
        for r in range(int((n - 1) / 2)):
            for c in range(int((n + 1) / 2)):
                row, col = (2 * r) + 1, 2 * c
                corners = get_corners(row, col, n)
                values = [inter_content[int((a + 1) * (offset + 1) - 1), int((b + 1) * (offset + 1) - 1)]
                          for a, b in corners]
                values = [val for val in values if val != -1]
                inter_content[(row + 1) * (offset + 1) - 1, (col + 1) * (offset + 1) - 1] = -1 if len(values) == 0 \
                    else average(values)

    # cut out extra borders (0.5 channel width) on all sides
    offset = int(inter_content.shape[0] / (content.shape[0] + 1) / 2)
    new_inter_content = np.zeros((new_dim - (2 * offset), new_dim - (2 * offset)))
    for row in range(offset, new_dim - offset):
        for col in range(offset, new_dim - offset):
            new_inter_content[row - offset, col - offset] = inter_content[row, col]
    real_channel_indices = np.array(real_channel_indices)
    real_channel_indices -= offset
    real_electrode_indices -= offset

    s = new_inter_content.shape[0]
    array_inds = [(i, s - j - 1) for i in range(s) for j in range(s)]

    # order real channel inds from 1 to 64
    r_inds = []
    tmp = real_channel_indices.reshape((8, 8, 2))
    for r in range(7, -1, -1):
        for c in range(7, -1, -1):
            r_inds.append(tmp[c, r, :])
    real_channel_indices = np.array(r_inds)

    new_inter_content_flat = np.array(new_inter_content.flat)

    return new_inter_content_flat, array_inds, real_channel_indices, real_electrode_indices


def calculate_correlation_maps_pool(paths_spikes, params, tag='', frames_path='', return_results=False):
    """
    Main function used to calculate correlation maps from signals (MUAe, LFP) or spikes (MUA, nLFP)
    and save them into path defined by parameters.

    :param paths_spikes: paths to spikes or signals
    :param params: analysis parameters
    :param tag: signal or spikes type (MUA, MUAe, LFP, nLFP)
    :param frames_path: path for precomputed frames
    :param return_results: True if results should also be returned
    :return:
    """
    calc_frames = False
    if len(frames_path) > 0:
        try:
            # load precomputed frames
            print('Trying to load frames from ', frames_path)
            with open(frames_path, "rb") as resfile:
                frs_data = pickle.load(resfile)
            frames = np.array(frs_data['frames'])
        except:
            print('Will calculate frames first.')
            calc_frames = True
    else:
        calc_frames = True

    if 'MUAe' in tag:
        frames = get_frames_MUAe(paths_spikes, params)
    elif 'nLFP' in tag:
        frames = get_frames_nLFP(paths_spikes, params)
    elif 'LFP' in tag:
        frames = get_frames_lfp(paths_spikes, params)
    elif calc_frames:
        frames = []
        rates = []
        durs = []
        times = []
        for pnr, path in enumerate(paths_spikes):
            frs, rts, spike_counts, duration, ts = get_frames(path, params, tag)
            if len(frs) == 0:
                continue
            frames.extend(frs)
            rates.append(rts)
            durs.append(duration)
            bs = ts[1] - ts[0]
            if len(times) > 0:
                ts = times[-1] + bs + ts
            times.extend(ts.tolist())
        frames = np.array(frames)  # frames x channels

        frames_path = os.path.join(params['results_path'], 'frames_{}_tf{}.pkl'.format(tag, params['thr_factor']))
        print(f'Saving frames to {frames_path}.')
        with open(frames_path, "wb") as a_file:
            pickle.dump({'frames': frames,
                         'paths': paths_spikes,
                         'durations': durs,
                         'times': np.array(times)}, a_file)  # here frs -> frames

        rates_path = os.path.join(params['results_path'], 'rates_{}_tf{}.pkl'.format(tag, params['thr_factor']))
        print(f'Saving rates to {rates_path}.')
        with open(rates_path, "wb") as a_file:
            pickle.dump({'rates': rates}, a_file)  # here frs -> frames

    print('Frames have shape:{}'.format(frames.shape))
    res = {'correlation_maps': [],
           'seed_channels': [],
           'rates': [],
           'interpolated_correlation_maps': [],
           'cor_vs_dist': []}

    for seed_ch_ix in range(frames.shape[1]):
        seed_ch = seed_ch_ix + 1
        res['seed_channels'].append(seed_ch)

        # correlate one seed channel with all other channels
        cors = []
        if len(frames) > 0:
            for chn in range(frames.shape[1]):
                # test whether array is constant
                if np.abs(np.min(frames[:, chn]) - np.max(frames[:, chn])) < 10e-6:
                    cors.append(0.)
                else:
                    cor = scipy.stats.pearsonr(frames[:, seed_ch_ix], frames[:, chn])[0]

                    # calculate distance between channels
                    coords1 = helper.get_coords_from_channel_index(seed_ch_ix, tag, params)
                    coords2 = helper.get_coords_from_channel_index(chn, tag, params)
                    dist = np.linalg.norm(np.array(coords1) - np.array(coords2))
                    res['cor_vs_dist'].append([cor, dist])
                    cors.append(cor)

            # cors[seed_ch_ix] = 0 # remove the self-correlation
            # with spikes this point is an outlier and affect the results a lot.

            cors = np.array(cors)
            cors = np.nan_to_num(cors)

        res['correlation_maps'].append(cors)
        # res['spike_counts'].append(np.sum(frames[:, seed_ch_ix]))

        cormap_array = np.zeros((8, 8))

        for chnix, cor in enumerate(cors):
            cx, cy = helper.get_coords_from_channel_index(chnix, tag, params)
            cormap_array[int((cx + 1400.) / 400.), int((cy + 1400.) / 400.)] = cor

        intermap, array_inds, real_chn_inds, real_el_inds = interpolate_data(cormap_array, params, tag=tag)
        if seed_ch_ix == 0:
            if 'MUAe' in tag or ('LFP' in tag):
                fr_tag = frames_path.split("/")[1]
                monkey = "L" if "monkey_L" in fr_tag else "A"
                arrayid = 0
                for i in range(16):
                    if f"{monkey}{i}" in fr_tag:
                        arrayid = i
            else:
                arrayid = int(paths_spikes[0].split("/")[-2].split('_')[1][1:])
                monkey = paths_spikes[0].split("/")[-2].split('_')[1][0]
            params['data_folder'] = params['data_folder_monkey']
            intermap, bad_channels = helper.remove_bad_channels(intermap, real_chn_inds, monkey, arrayid, params,
                                                                tag)
        res['interpolated_correlation_maps'].append(intermap)

    # remove bad channels from all interpolated maps
    inter_map_full = np.array(res['interpolated_correlation_maps'])
    inds = np.where(inter_map_full[0] == params['bad_channel_value'])
    inter_map_full[:, inds] = params['bad_channel_value']
    res['interpolated_correlation_maps'] = inter_map_full.T

    # res['interpolated_correlation_maps'] = np.array(res['interpolated_correlation_maps']).T
    res['array_inds'] = array_inds
    res['real_channel_inds'] = real_chn_inds
    res['real_electrode_inds'] = real_el_inds
    res['bad_channels'] = bad_channels

    res['correlation_maps'] = np.array(res['correlation_maps'])
    res['correlation_values'] = np.array(res['correlation_maps'].flat)
    res['correlation_maps'][np.where(np.abs(res['correlation_maps']) < params['correlation_threshold'])] = 0.

    if return_results:
        return res
    else:
        print('Saving to:')
        respath = os.path.join(params['results_path'], f"correlation_maps_pool_{tag}.pkl")
        print(respath)
        with open(respath, "wb") as a_file:
            pickle.dump(res, a_file)


def PCA_map(params, tag, data={}, return_results=False):
    """
    Main function used to calculate spontaneous map from interpolated correlation maps.

    :param params: analysis parameters
    :param tag: {monkey}{array_id}_{method}
    :param data: correlation maps data
    :param return_results: True if results should also be returned
    :return:
    """
    from sklearn.decomposition import PCA

    if len(data.keys()) == 0:
        respath = os.path.join(params['results_path'], f"correlation_maps_pool_{tag}.pkl")
        print('Loading data from:')
        print(respath)

        try:
            with open(respath, "rb") as resfile:
                data = pickle.load(resfile)
        except:
            print('Need to calculate correlation maps first.')
            exit()
    else:
        print('Data passed as an argument.')

    # locations = data['locations']
    array_inds = np.array(data['array_inds'])

    Cmaps = data['interpolated_correlation_maps']
    map_dim = int(np.sqrt(Cmaps.shape[0]))
    len_Cmaps = len(Cmaps)

    # extract tunned channels only
    ids_good = np.where(Cmaps[:, 0] > params['bad_channel_value'])[0]
    array_inds = array_inds[ids_good]
    Cmaps = Cmaps[ids_good]

    pca = PCA(n_components=10)
    pca.fit(Cmaps)
    points_pca = pca.transform(Cmaps)

    print('Variance explained ratio:')
    print(pca.explained_variance_ratio_)
    print('sum:', np.sum(pca.explained_variance_ratio_))

    # circular coloring
    points_2d = points_pca[:, params['remove_PCA_dims']:params['remove_PCA_dims'] + 2]

    center = points_2d.mean(axis=0)
    p = points_2d - center
    angle = np.arctan(p[:, 0] / p[:, 1])

    angle[p[:, 1] < 0] += np.pi
    angle -= angle.min()

    community_labels = angle / 2.

    pca_map = np.zeros((map_dim, map_dim)) + params['bad_channel_value']
    cor_map = np.zeros((map_dim, map_dim)) + params['bad_channel_value']

    community_labels_all = np.zeros(len_Cmaps) + params['bad_channel_value']
    community_labels_all[ids_good] = community_labels

    for i in range(len(Cmaps)):
        pca_map[array_inds[i][0], array_inds[i][1]] = community_labels[i]  # angle[i]
        # save example correlation maps for plotting
        if 'L11' in tag:
            cor_map[array_inds[i][0], array_inds[i][1]] = Cmaps[i, 27]
        elif 'L13' in tag:
            cor_map[array_inds[i][0], array_inds[i][1]] = Cmaps[i, 25]
        elif 'patient2' in tag:
            cor_map[array_inds[i][0], array_inds[i][1]] = Cmaps[i, 81]

    if params['back_interpolate']:
        real_vals = np.full((8, 8), -2.)
        inds = data["real_channel_inds"]
        ix = 0
        for r in range(7, -1, -1):
            for c in range(7, -1, -1):
                real_vals[c, r] = pca_map[inds[ix, 0], inds[ix, 1]]
                ix += 1

        real_vals = real_vals / np.pi * 180
        real_vals[np.where(real_vals < 0)] = -1  # -1 is untunned value for OP maps
        inter = interpolate_data(real_vals, params, "monkey_orientation")[0].reshape((65, 65))
        inter[np.where(pca_map < 0)] = -2.
        pca_map = inter

        community_labels_all = np.flip(pca_map, axis=1).reshape(4225) / 180 * np.pi

    res = {'pca_array': pca_map,
           'cor_array': cor_map,
           'modularity': -1.,
           'points_pca_plane': points_2d,
           'points_pca': points_pca,
           'points_labels': community_labels_all,
           'real_channel_inds': data['real_channel_inds'],
           'real_electrode_inds': data['real_electrode_inds'],
           'spontaneous_map_flat': community_labels,
           'bad_channels': data['bad_channels'],
           'rates': data['rates'],
           'explained_variance': pca.explained_variance_ratio_,
           'cor_vs_dist': data['cor_vs_dist']}  # modularity}

    if return_results:
        return res
    else:
        respath = os.path.join(params['results_path'], f"pca_communities_{tag}.pkl")
        with open(respath, "wb") as a_file:
            pickle.dump(res, a_file)
