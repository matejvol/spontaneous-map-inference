"""
Testing
This script contains functions for
1) data transformations
    - shift_invert()
    - best_distances()
2) generating controls
    - permutate_OP_map()

Authors: Matěj Voldřich, Karolína Korvasová
"""
import neo
import numpy as np
import helper


def permutate_OP_map(orientations: np.array, size=1, empty_val=-1):
    """
    Permutate orientation maps $size times and return result as numpy array.
    Untunned channels are kept in place.

    :param orientations: orientations 1D array
    :param size: number of permutations
    :param empty_val: value used to indicate untunned channels excluded from permutation
    :return: permutated orientations
    """
    original_indexes = [i for i in range(len(orientations)) if orientations[i] != empty_val]
    original_values = [val for val in orientations if val != empty_val]

    if size == 1:
        perm_indexes = np.random.permutation(original_indexes)
        perm_orientations = np.full(len(orientations), empty_val)
        for i, ix in enumerate(perm_indexes):
            perm_orientations[ix] = orientations[original_indexes[i]]
        return perm_orientations
    else:
        perm_orientations = np.zeros((size, len(original_values)))
        for p in range(size):
            perm_orientations[p, :] = np.random.permutation(original_values)
        # fill in untunned channels
        new_orientations = np.full((size, len(orientations)), empty_val)
        untunned_indexes = np.where(np.array(orientations) == empty_val)[0]
        if len(untunned_indexes) > 0:
            new_orientations[:, 0:untunned_indexes[0]] = perm_orientations[:, 0:untunned_indexes[0]]
            for i, ix in enumerate(untunned_indexes):
                prev_ix = untunned_indexes[i - 1]
                new_orientations[:, prev_ix + 1:ix] = perm_orientations[:, prev_ix - i + 1:ix - i]
            last_ix = untunned_indexes[-1]
            new_orientations[:, last_ix + 1:] = perm_orientations[:, (last_ix - len(untunned_indexes) + 1):]
        else:
            new_orientations = perm_orientations
        return new_orientations


def shift_invert(data: np.array, vlim: tuple, n_sifts=10, invert=True, empty_val=-1):
    """
    Shift and invert circular data to find best match to evoked map.

    :param data: 1D data array
    :param vlim: (min value, max value)
    :param n_sifts: number of shifts
    :param invert: invert data if True for each shift
    :param empty_val: value used to indicate untunned channels
    :return: all shift/inverted combinations in np array, transformation
    """
    vmin, vmax = vlim
    empty_inds = np.where(data == empty_val)
    results = np.zeros((n_sifts * (2 if invert else 1), len(data)))
    transforms = np.zeros((n_sifts * (2 if invert else 1),2))
    results[0, :] = data
    if invert:
        results[1, :] = vmax - results[0, :]
    step = (vmax - vmin) / n_sifts
    for i in range(1, n_sifts):
        if invert:
            results[2 * i, :] = results[2 * i - 2, :] + step  # shift
            results[2 * i, np.where(results[2 * i, :] > vmax)] -= vmax
            results[2 * i + 1, :] = vmax - results[2 * i, :]  # invert
            transforms[2 * i + 1,0] = invert
            transforms[2 * i + 1,1] =(i+1)*step
        else:
            results[i, :] = results[i-1, :] + step
            transforms[i,0] = invert
            transforms[i,1] =(i+1)*step
    results[np.where(results > vmax)] -= vmax
    results[:, empty_inds] = empty_val
    return results, transforms


def best_distances(data, reference, vmax, feature=np.mean, return_index=False):
    """
    Calculate best score for data shifts/inversions.

    :param data: shifted or inverted data as 2D array
    :param reference: reference data as 1D array
    :param vmax: maximum circular value
    :param feature: statistical feature to be minimalized
    :param return_index: return index for best shift/inversion
    :return: distances for best shift/inversion
    """
    distances = np.zeros((2, data.shape[0], data.shape[1]))
    distances[0, :, :] = np.abs(data - reference)
    joined = np.array([data, np.full_like(data, reference)])
    distances[1, :, :] = np.abs(vmax - np.max(joined, axis=0) + np.min(joined, axis=0))
    circular_distances = np.min(distances, axis=0)
    scores = np.array([feature(circular_distances[i, :]) for i in range(circular_distances.shape[0])])
    best_distance = circular_distances[np.where(scores == np.min(scores))[0], :][0]
    if return_index:
        return best_distance, np.where(scores == np.min(scores))[0][0]
    return best_distance


def remove_untunned_channels(interpolation_result, orientations, params, monkey, Array_ID, fill_value=-1):
    """
    Replace values in untunned channels with $fill_value for interpolated map

    :param interpolation_result: tuple result from interpolate_date
    :param orientations: 2D orientations (not interpolated)
    :param params: parameters (interpolation factor
    :param Array_ID: array ID
    :param fill_value: fill value
    :return: 2D interpolated map
    """
    REMOVE_LOW_FR = False
    ifc = params["interpolation_factor"]
    offset = 2**(ifc-1)
    interpolated_data, _, r_inds = interpolation_result
    if REMOVE_LOW_FR:
        interpolated_data = remove_low_FR_channels(interpolation_result, monkey, Array_ID, params, 0.1)
    shape = int(np.sqrt(len(interpolated_data)))
    interpolated_data.resize((shape, shape))
    orientations = np.array(orientations.flat)

    # transform real inds
    tmp = r_inds.reshape((8, 8, 2))
    r_inds = []
    for r in range(7, -1, -1):
        for c in range(7, -1, -1):
            r_inds.append(tmp[c, r, :])
    r_inds = np.array(r_inds)

    for i in range(len(orientations)):
        if orientations[i] == -1:
            y, x = r_inds[i]
            interpolated_data[y-offset:y+offset+1, x-offset:x+offset+1] = fill_value
    return interpolated_data


def get_firing_rate(monkey, Array_ID, params):
    data_path = params["data_folder_monkey"]
    th = int(10*params["thr_factor"])
    _, file_names = helper.find_recordings(f"{data_path}/spikes_tf{th}_no_synchrofacts/Spontaneous_{monkey}{Array_ID}/",
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


def remove_bad_channels_OP_map(opmap, spontmap, params):

    ids = np.where(spontmap<-0.5)[0]
    opmap[ids] = params['bad_channel_value']

    return opmap


def remove_low_FR_channels(interpolation_result, monkey, array_id, params, FR_th, fill_value=-1):
    """

    :param interpolation_result:
    :param array_id:
    :param params:
    :param FR_th:
    :param fill_value:
    :return:
    """
    # calculate FRs
    firing_rates = get_firing_rate(monkey, array_id, params)

    # remove low FR channels
    ifc = params["interpolation_factor"]
    offset = 2 ** (ifc - 1)
    layout_flat = np.array(helper.get_electrode_layout(params, "monkey").flat)
    interpolated_data, _, r_inds = interpolation_result
    shape = int(np.sqrt(len(interpolated_data)))
    interpolated_data.resize((shape, shape))
    for i in range(len(layout_flat)):
        chn = layout_flat[i]
        if firing_rates[chn - 1] < FR_th:
            y, x = r_inds[i]
            interpolated_data[y - offset:y + offset + 1, x - offset:x + offset + 1] = fill_value
    # print(f"Removed {len(np.where(firing_rates < FR_th)[0])} channels with firing rate < {FR_th}.")
    return np.array(interpolated_data.flat)
