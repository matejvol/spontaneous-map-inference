"""
Helper
This script contains functions for
1) manipulating files
    - ensure_dir_exists()
    - find_recordings()
    - find_MUAe_recordings()
    - find_LFP_recordings()
2) loading signal and array metadata
    - get_metadata()
    - extract_signal_eyes_closed()
    - Roelfsema_metadata()
    - remove_bad_channels()
    - get_array_mean_SNR()
    - get_firing_rate()
    - get_Xing_orientation_tunning()
    - get_electrode_layout()
    - get_Roelfsema_chn_loc()
    - get_coords_from_channel()
    - get_coords_from_channel_index()
    - get_channels_without_OP()

Authors: Karolína Korvasová, Matěj Voldřich
"""
import elephant
import neo
import numpy as np
import quantities as pq
import datetime
import os
import pandas as pd
import scipy
import pickle


def ensure_dir_exists(dirpath):
    if not os.path.isdir(dirpath):
        print('Creating', dirpath)
        os.makedirs(dirpath)


def find_recordings(dirpath, keywords):
    '''
    Go through the files in the folder specified by dirpath and filter out
    files.

    arguments:
    -----------
    dirpath: path to the directory with data (string)

    returns:
    -----------
    subdir paths: list of strings, paths to the files with data
    subdirs_filt: list of strings, filenames
    '''
    subdirs = os.listdir(dirpath)
    subdirs_filt = []
    subdir_paths = []

    # exclude files containing exclude
    # for subdir in subdirs:
    #     for exc in exclude:
    #         if exc in subdir:
    #             subdirs.remove(subdir)

    for subdir in subdirs:
        key_in = True
        for keyword in keywords:
            if keyword not in subdir:
                key_in = False
        if key_in:
            subdirs_filt.append(subdir)
            subdir_paths.append(os.path.join(dirpath, subdir))
        else:
            print('No required data in '+subdir)


    return subdir_paths, subdirs_filt


def find_MUAe_recordings(monkey, array, params):
    head_folder = params['data_folder_monkey']
    NSP_ID = Roelfsema_metadata(f"select NSP_ID from mapping where Array_ID={array}")[0][0]
    dates = {'L': ['090817', '100817', '250717'],
             'A': ['140819', '150819', '160819']}
    paths = [f"{head_folder}/V1_V4_1024_electrode_resting_state_data/data/"
             f"{monkey}_RS_{date}/MUAe/NSP{NSP_ID}_array{array}_MUAe.nix" for date in dates[monkey]]
    return paths


def find_LFP_recordings(monkey, array, params):
    head_folder = params['data_folder_monkey']
    NSP_ID = Roelfsema_metadata(f"select NSP_ID from mapping where Array_ID={array}")[0][0]
    dates = {'L': ['090817', '100817', '250717'],
             'A': ['140819', '150819', '160819']}
    paths = [f"{head_folder}/V1_V4_1024_electrode_resting_state_data/data/"
             f"{monkey}_RS_{date}/LFP/NSP{NSP_ID}_array{array}_LFP.nix" for date in dates[monkey]]
    return paths


def get_metadata(path, params):
    dates = ["140819", "150819", "160819",  # A_RS
             "250717", "090817", "100817"]  # L_RS
    for i in range(len(dates)):
        date = dates[i]
        if date in path:
            monkey = "A" if i < 3 else "L"
            break
    metadata_path = f"{params['data_folder_monkey']}/V1_V4_1024_electrode_resting_state_data/" \
                    f"metadata/epochs/epochs_{monkey}_RS_{date}.csv"
    metadata = pd.read_csv(metadata_path)
    return metadata


def extract_signal_eyes_closed(path, params, fs):
    """
    Extract analog signals where monkey had eyes closed for
    at least given interval in params['min_eyes_closed']
    """
    metadata = get_metadata(path, params)

    seg = neo.NixIO(path, "ro").read_block().segments[0]
    an_sigs = np.array(seg.analogsignals[0].magnitude)
    new_sigs = [[] for _ in range(64)]
    t_start, t_stop, duration, state = metadata["t_start"], metadata["t_stop"], metadata["dur"], metadata["state"]
    for i in range(len(state)):
        if state[i] == "Closed_eyes" and duration[i] >= params['min_eyes_closed']:
            start_ix = int(t_start[i] * fs)
            stop_ix = int(t_stop[i] * fs)
            for j in range(64):
                new_sigs[j].extend((an_sigs[start_ix:stop_ix, j]).tolist())
    # seg.analogsignals[0] = neo.core.AnalogSignal(np.array(new_sigs), units='uV', sampling_rate=fs * pq.Hz)
    return np.array(new_sigs).T


def calculate_nLFP(signal, thr_factor):
    bl = neo.Block()
    bl.segments.append(neo.Segment())
    thr = thr_factor * np.std(signal)
    if thr_factor > 0:
        sign_ext = 'above'
    else:
        sign_ext = 'below'
    # extract peaks
    for i in range(signal.shape[1]):
        st = elephant.spike_train_generation.peak_detection(
            signal[:, i],
            threshold=np.array(thr) * signal.units,
            sign=sign_ext)
        bl.segments[0].spiketrains.append(st)
    return bl


def load_raw(path, monkey='', arrayid=0):
    '''
    Load data from a file specified by path.

    Arguments:
    ------------
    path:  string

    Returns:
    ------------
    seg: neo segment
    '''
    reader = neo.io.BlackrockIO(filename=path)


    seg = reader.read_segment()
    print(f'Loaded {path}.')
    if len(monkey)>0:
        # Is this a better way?
        # Or using odml?
        # channel_ids = seg.analogsignals[-1].array_annotations['channel_ids'].copy()

        path_map = f'/projects/Roelfsema_data/V1_V4_1024_electrode_resting_state_data/metadata/experimental_setup/channel_area_mapping_{monkey}.csv'
        electrode_list = pd.read_csv(path_map)
        print(electrode_list)
        print(arrayid)
        ids_array = np.where(electrode_list['Array_ID']==arrayid)[0]
        # find the index where the current nsp starts
        nsp = electrode_list['NSP_ID'][ids_array[0]]
        ids_nsp = np.where(electrode_list['NSP_ID']==nsp)[0]
        nsp_offset = np.min(ids_nsp)

        anasig = seg.analogsignals[0]
        electrode_nrs = electrode_list['within_array_electrode_ID'][ids_array].values
        perm = np.argsort(electrode_nrs)

        import pandas as pd
        import numpy as np
        arrayid = 11
        monkey = 'L'
        path_map = f'/projects/Roelfsema_data/V1_V4_1024_electrode_resting_state_data/metadata/experimental_setup/channel_area_mapping_{monkey}.csv'
        electrode_list = pd.read_csv(path_map)
        ids_array = np.where(electrode_list['Array_ID'] == arrayid)[0]
        # find the index where the current nsp starts
        nsp = electrode_list['NSP_ID'][ids_array[0]]
        ids_nsp = np.where(electrode_list['NSP_ID'] == nsp)[0]
        nsp_offset = np.min(ids_nsp)
        electrode_nrs = electrode_list['within_array_electrode_ID'][ids_array].values
        perm = np.argsort(electrode_nrs)

        correct_channels = ids_array[perm]-nsp_offset


        print('ids_array', ids_array)
        print('electrode nrs', electrode_nrs)
        print('anasig shape', anasig.shape)
        print('perm', perm)
        seg_new = neo.Segment()
        for i in ids_array[perm]-nsp_offset:  # anasig.shape[1]):
            sig = np.squeeze(anasig[:, i])
            sig_new = neo.AnalogSignal(sig * pq.mV,
                                       unit=pq.mV,
                                       sampling_rate=anasig.sampling_rate,
                                       t_start=anasig.t_start,
                                       t_stop=anasig.t_stop)
            seg_new.analogsignals.append(sig_new)

    else:
        anasig = seg.analogsignals[0]

        seg_new = neo.Segment()
        for i in range(96): #anasig.shape[1]):
            # some files contain 98 or 99 channels, but the last ones are not real data
            # they contain very high values
            # not clear what this is, neurotic ignores these channels
            sig = np.squeeze(anasig[:,i])
            sig_new = neo.AnalogSignal(sig*pq.mV,
                            unit=pq.mV,
                            sampling_rate=anasig.sampling_rate,
                            t_start=anasig.t_start,
                            t_stop=anasig.t_stop)
            seg_new.analogsignals.append(sig_new)

    seg_new.file_origin = path
    seg_new.file_datetime = datetime.datetime
    print('Number of analog signals:', len(seg_new.analogsignals))

    return seg_new


def Roelfsema_metadata(query: str):
    """
    Get data from Roelfsema metadata database:

        Table "mapping": {Electrode_ID,NSP_ID,within_NSP_electrode_ID,Array_ID,within_array_electrode_ID,Area, orientations_index}

        Table "SNR": {Electrode_ID,date,snr,peak_response,baseline_avg,baseline_std,response_onset_timing,monkey}
            date: 2017-07-25, 2017-08-10, 2017-08-09

        Table "impedance" {Electrode_ID,Array_ID,Aug10,Aug18}

        Table "impedance_A" {Electrode_ID,Array_ID,impedance}

    :param query: SQL query
    :return: query result as a list of tuples
    """
    import sqlite3
    connection = sqlite3.connect("monkey_data.db")
    cursor = connection.cursor()
    try:
        result = cursor.execute(query).fetchall()
    except:
        print(f"Invalid query: '{query}'")
        return None
    connection.close()

    return result


def convolve_signal(x, ts, params, tau=-1):
    '''
    Computes a convolution of x with the synaptic kernel exp(-t/tau_syn)/tau_syn

    arguments:
    x: array
    T: float
        simulation time

    return:
    conv_time: array
       (number of time steps x dimension of x)
    '''
    if tau<0:
        tau = params['imaging_kernel']

    h = np.exp(-ts/tau)/tau
    conv_time = np.zeros(x.shape)

    # import ipdb
    # ipdb.set_trace()

    if len(x.shape)>1:
        for i in range(x.shape[1]):
            conv_time[:,i] = np.real(np.fft.ifft(np.fft.fft(x[:,i])*np.fft.fft(h)))
    else:
        conv_time = np.real(np.fft.ifft(np.fft.fft(x)*np.fft.fft(h)))

    ## TODO: fft causes artefacts at the boundaries.
    ##       Either use zero padding or use a different
    ##       method for computing the convolution.

    return conv_time


def remove_bad_channels(interpolated_map, r_inds, monkey, Array_ID, params, tag="monkey"):
    """
    Replace values for bad channels (no OP, low SNR, low FR) in interpolated map with special "bad_channel_value".

    :param interpolated_map: 1D interpolated map
    :param r_inds: real channel indices
    :param Array_ID:
    :param params:
    :param tag:
    :return: 1D interpolated map with replaced bad channel channels
    """
    n = int(np.sqrt(interpolated_map.shape[0]))
    interpolated_map = interpolated_map.reshape((n, n))
    bad_channels = []  # (indexes)
    # remove untunned channels
    if params['only_oriented']:
        bad_channels.extend(get_channels_without_OP(monkey, Array_ID, params) - 1)
        print('No orientation preference:', np.sort(bad_channels) + 1)

    # remove channels with SNR below threshold
    snr_th = params['snr_th']
    if snr_th > 0:
        snr = get_array_mean_SNR(Array_ID, monkey)
        bad_channels.extend(np.where(snr < snr_th)[0])
        print('Channels with low SNR:', np.where(snr < snr_th)[0] + 1)

    # remove channels with FR below threshold
    fr_th = params['fr_th']
    if fr_th > 0:
        rates_path = os.path.join(params['results_path'], 'rates_{}_tf{}.pkl'.format(tag, params['thr_factor']))
        with open(rates_path, "rb") as resfile:
            rates_data = pickle.load(resfile)
        array_fr = np.array(rates_data['rates'])
        bad_channels.extend(np.where(array_fr < fr_th)[0])

    bad_channels = np.unique(bad_channels)
    # replace channel values
    fill_value = params['bad_channel_value']
    ifc = params["interpolation_factor"]
    offset = 2 ** (ifc - 1)
    for chn_ix in bad_channels:
        x, y = r_inds[chn_ix, :]
        interpolated_map[int(x-offset):int(x+offset+1),
                         int(y-offset):int(y+offset+1)] = fill_value

    return np.array(interpolated_map.flat), bad_channels


def get_array_mean_SNR(Array_ID, monkey='L'):
    result = Roelfsema_metadata("select snr from mapping inner join SNR on mapping.Electrode_ID=SNR.Electrode_ID "
                                f"where Array_ID={Array_ID} and monkey='{monkey}' order by within_array_electrode_ID")
    result = np.array(result)
    n_measurements = int(result.shape[0] / 64)
    snr = np.zeros(64)
    for i in range(64):
        snr[i] = np.mean(result[i*n_measurements:(i+1)*n_measurements])
    return snr


def get_firing_rate(monkey, Array_ID, params):
    """
    Calculate firing rate for MUA spikes
    :param Array_ID:
    :param params: analysis parameters
    :return: FR
    """
    data_path = params["data_folder_monkey"]
    th = int(params["thr_factor"])
    _, file_names = find_recordings(f"{data_path}/spikes_tf{th}_no_synchrofacts/Spontaneous_{monkey}{Array_ID}/",
                                           ["nix"])
    duration = 0
    spikes_all = np.zeros(64)
    for file_name in file_names:
        reader = neo.NixIO(f"{data_path}/spikes_tf{th}_no_synchrofacts/Spontaneous_{monkey}{Array_ID}/{file_name}", "ro")
        bl = reader.read_block()
        t_stop = int(bl.segments[0].spiketrains[0].t_stop)
        duration += t_stop
        spikes = np.array([len(bl.segments[0].spiketrains[i]) for i in range(64)])
        spikes_all += spikes
    firing_rates = spikes_all / duration
    return firing_rates


def get_Xing_orientation_tunning(monkey: str, array_n: int, params: dict, all=False, round=False, round_classes=12):
    """
    Load orientation preference for given array.

    :param array_n: Array ID (1-16)
    :param params: parameters
    :param all: Will only return good orientations if False
    :param round: Round orientations to the nearest 15-fold
    :return orientations: Orientations (0° - 180°) for all channels as 2d array
    """
    from mat4py import loadmat
    # source paths
    data_path = params['data_folder_monkey']
    all_orientations_path = data_path + f"/Xing/channels_preferred_orientations/all_channels_preferred_orientations.mat"
    good_orientations_path = data_path + f"/Xing/good_channels_preferred_orientations_{monkey}.mat"
    layout = get_electrode_layout(params, "monkey")

    if all:
        orientation_path = all_orientations_path
        key = "allPrefOri1024"
        widths = np.squeeze(loadmat(orientation_path)['allWidth1024'])
        pvas = np.squeeze(loadmat(orientation_path)['allPVA1024'])
    else:
        orientation_path = good_orientations_path
        key = "allPrefOri"


    orientations = loadmat(orientation_path)[key]

    content = np.zeros((8, 8)) - 1
    indexes = Roelfsema_metadata(f"select NSP_ID, within_NSP_electrode_ID from mapping where Array_ID={array_n} "
                                 f"order by within_array_electrode_ID")
    NSP = indexes[0][0]
    indexes = [ix[1] for ix in indexes]

    for row in range(8):
        for col in range(8):
            chn = layout[row, col]
            index = (NSP - 1) * 128 + indexes[chn-1] - 1
            if not all:
                content[row, col] = orientations[index][0]
            else:
                if (widths[index]>params['tuning_width_thr']) and (pvas[index]>params['pva_thr']):
                    content[row, col] = orientations[index][0]

            if round and content[row, col] != -1:
                val = np.round(content[row, col] / 180 * round_classes)
                if val == 0:
                    val = round_classes
                content[row, col] = val * 180 / round_classes

    return content


def get_electrode_layout(params, tag):
    eldist = 400.  # unit um
    path_map = '{}/V1_V4_1024_electrode_resting_state_data/metadata/experimental_setup/elec_position_in_array.csv'.format(
        params['data_folder'])
    electrode_map = pd.read_csv(path_map)
    x_col, y_col, chn_col = "x (um)", "y (um)", "Elec_ID"

    x_size = len(np.unique(electrode_map[x_col])[~np.isnan(np.unique(electrode_map[x_col]))])
    y_size = len(np.unique(electrode_map[y_col])[~np.isnan(np.unique(electrode_map[y_col]))])
    offset = [min(electrode_map[x_col]), min(electrode_map[y_col])]

    layout = np.full((y_size, x_size), -1)
    for x, y, chn in zip(electrode_map[x_col], electrode_map[y_col], electrode_map[chn_col]):
        row = int((y - offset[1]) / eldist)
        col = int((x - offset[0]) / eldist)
        if pd.isna(chn):
            continue
        layout[row, col] = int(chn)
    return layout


def load_spikes(path, tag=''):
    '''
    Load spiking activity from different formats.

    Arguments:
    -----------
    path: string, path to data

    Returns:
    -----------
    bl: neo block with spikes
    '''
    if path[-3:]=='nix':
        reader = neo.io.NixIO(filename=path, mode='ro')
        bl = reader.read_block() #load_waveforms=True)
    elif path[-3:]=='nev':
        reader = neo.io.BlackrockIO(filename=path)
        bl = reader.read_block()
    elif path[-3:]=='mat':
        if 'gratings' in tag:
            spikes = np.squeeze(scipy.io.loadmat(path)['data']['EVENTS'][0][0])
            spikes_conc = []
            t_stop = 0.
            for unit in spikes:
                spikes_unit = []
                for orient in range(unit.shape[0]):
                    for trial in range(unit.shape[1]):
                        st = unit[orient, trial]
                        spikes_unit.extend(st)
                        if len(st)>0:
                            t_stop += st[-1]
                spikes_conc.append(spikes_unit)
            spikes = np.squeeze(np.sort(np.array(spikes_conc)))

        else:
            spikes = np.squeeze(scipy.io.loadmat(path)['data']['EVENTS'][0][0][0])

        bl = neo.Block()
        seg = neo.Segment()
        bl.segments.append(seg)

        if 'gratings' not in tag:
            t_stop = 0. #np.array([0.])*pq.s
            for st in spikes:
                if st[-1]>t_stop:
                    t_stop = st[-1]

        for st in spikes:
            seg.spiketrains.append(neo.SpikeTrain(np.squeeze(st)*pq.s,
                        t_stop = t_stop*pq.s
            ))
    elif path[-3:]=='npy':

        data = np.load(path, allow_pickle=True)
        data_list = data.tolist()
        sr = 30000. # float(data_list['SamplingRate'])

        # if len(data_list)>100:
        #     # the case when spikes from the whole nsp was extracted in the monkey
        #     inds = get_indices_from_nsp(monkey, arrayid)
        #     data = np.array(data_list)[inds]
        #     data_list = data.tolist()

        if "monkey" in tag:
            spikes = [[] for i in range(64)]
        else:
            spikes = [[] for i in range(96)]
        for chn, spike, goodspike in zip(data_list['ChannelID'],
                                         np.array(data_list['TimeStamps']) / sr,
                                         data_list['UnitID']):
            if goodspike>0:
                spikes[chn - 1].append(spike) # indexed from 1


        bl = neo.Block()
        seg = neo.Segment()
        bl.segments.append(seg)


        t_stop = 2000.
        for st in spikes:
            st = np.array(st)
            # assert st[-1]<t_stop
            seg.spiketrains.append(neo.SpikeTrain((st)*pq.s,
                        t_stop = t_stop*pq.s # remove if normal t_stop
            ))

    print(f'\n Loaded {path}.')
    # print('t_stop:', t_stop)
    # print('\n')


    return bl


def get_Roelfsema_chn_loc(raw_folder, channel_number):
    '''
    Get the location of the channel.
    Channels numbered from 1.
    '''
    import csv
    path = raw_folder + '/V1_V4_1024_electrode_resting_state_data/metadata/experimental_setup/elec_position_in_array.csv'

    with open(path) as csvfile:
        csvreader = csv.reader(csvfile)

        rows = []
        for row in csvreader:
                rows.append(row)

    # return float(rows[channel_number][1])+1400., float(rows[channel_number][2])+1400.
    return float(rows[channel_number][1]), float(rows[channel_number][2])


def get_coords_from_channel(channel_number):
    # channels numbered from 1

    electrodes_channels_coords_map = pd.read_pickle('metadata/electrode_mapping.pickle')
    x = electrodes_channels_coords_map[electrodes_channels_coords_map.Channels==channel_number].x.values[0]
    y = electrodes_channels_coords_map[electrodes_channels_coords_map.Channels==channel_number].y.values[0]

    return x,y


def get_coords_from_channel_index(chnix, tag, params):

    if "blind" in tag:
        cx, cy = get_coords_from_channel(
            channel_number=chnix + 1)
    elif ("Roelfsema" in tag) or ("monkey" in tag):
        cx, cy = get_Roelfsema_chn_loc(
            params['data_folder_monkey'], chnix + 1)
    elif "model" in tag:
        electrode_map = np.array(pd.read_pickle('data_folder/model_from_Tibor/neuron_positions.pkl')['positions'])
        cx, cy = electrode_map[chnix]
    return cx, cy


def get_channels_without_OP(monkey, arraynr, params):

    import scipy.io

    print('Array nr:', arraynr)
    path_folder = params['data_folder_monkey'] + '/Xing'
    pos = scipy.io.loadmat(path_folder + f'/good_channels_preferred_orientations_{monkey}.mat')
    path_map = path_folder + '/channel_area_mapping.mat'
    mapping = scipy.io.loadmat(path_map)
    assert len(np.unique(np.where(mapping['arrayNums'] == 16)[1]) == 1)
    ids_array = np.where(mapping['arrayNums'] == arraynr)
    nsp = ids_array[1][0]
    print('NSP:', nsp)
    ids_channels_nsp = ids_array[0]
    ids_mapping = nsp * 128 + ids_channels_nsp

    po_array = np.squeeze(pos['allPrefOri'][ids_mapping])
    # po_array = np.squeeze(pos['allPrefOri1024'][ids_mapping])
    # widths = np.squeeze(pos['allWidth1024'][ids_mapping])
    # pvas = np.squeeze(pos['allPVA1024'][ids_mapping])
    #
    # # sort preferred orientations based on the channel number
    perm = np.argsort(mapping['channelNums'][:, nsp][ids_channels_nsp])
    po_array_sorted = po_array[perm]
    po_array_final = po_array_sorted

    # widths_sorted = widths[perm]
    # pvas_sorted = pvas[perm]

    # only choose well tuned ones

    # ids_good = np.where((widths_sorted>params['tuning_width_thr']) & (pvas_sorted>params['pva_thr']))
    # po_array_final = np.zeros_like(po_array_sorted) - 1.
    # po_array_final[ids_good] = po_array_sorted[ids_good]

    exclude_chn = np.where(po_array_final < 0)[0] + 1

    return exclude_chn


def KDE(data, bandwidth=1):
    """
    Gaussian Kernel density estimation
    :param data: list or array of values
    :param bandwidth:
    :return: x, y (density)
    """
    from sklearn.neighbors import KernelDensity
    x = np.array(data)
    x.sort()
    X = x[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
    return x, np.exp(kde.score_samples(X))