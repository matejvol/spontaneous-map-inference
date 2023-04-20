"""
Test spontaneous map
This script is used to test statistical significance of a spontaneous map.

Arguments:
    1) {monkey}{Array ID}
    2) method
    3) unused
    4) scan id if running parameter search
    5) number of principal components to running parameter search

Authors: Matěj Voldřich, Karolína Korvasová
"""
import yaml
import pickle
import analysis
import scipy
import plotting
from testing import *
import sys
import os

if __name__ == "__main__":
    with open("params_paths.yml") as f:
        params = yaml.safe_load(f)
    with open("params_analysis.yml") as f:
        params.update(yaml.safe_load(f))
    with open("params_plotting.yml") as f:
        params.update(yaml.safe_load(f))

    if len(sys.argv)>4:
        id = sys.argv[4]
        pca = int(sys.argv[5])
        import pickle
        with open('parameter_space.pkl', "rb") as parspfile:
            pars_update = pickle.load(parspfile)
        print('Updating parameters.')
        print(pars_update['parameters'][int(id)])
        params.update(pars_update['parameters'][int(id)])
        params['remove_PCA_dims'] = pca


    params["data_folder"] = params["data_folder_monkey"]
    params.update({
        'results_path': f"{params['head_folder']}/testing",
        'figures_path': params['head_folder']
    })

    # SETUP
    array_n = int(sys.argv[1][1:])
    monkey = sys.argv[1][0]
    method = sys.argv[2]
    n_shifts = 40
    distance_feature = np.mean  # statistical feature to be minimized in possible shifts/inversions
    score = lambda data: distance_feature(data)  # distance_feature based score (to be maximized)

    helper.ensure_dir_exists(params['results_path'])
    helper.ensure_dir_exists(os.path.join(params['results_path'], method))

    # load spontaneous and orientation data
    ori = helper.get_Xing_orientation_tunning(monkey, array_n, params, all=False)
    ori_flat = ori.reshape(64)
    try: # ideally the tag should be passed
        path = f"{params['head_folder']}/pca_communities_monkey_{monkey}{array_n}_{method}_spont.pkl"
        with open(path, "rb") as f:
            spont_data = pickle.load(f)
    except:
        path = f"{params['head_folder']}/pca_communities_scan_monkey_{monkey}{array_n}_{method}_spont.pkl"
        with open(path, "rb") as f:
            spont_data = pickle.load(f)
    real_inds = spont_data["real_channel_inds"]
    spont_shape = spont_data["pca_array"].shape
    spont_data = spont_data["points_labels"].reshape(spont_shape).T
    spont_data = spont_data * 180 / np.pi
    spont_data[np.where(spont_data < 0)] = params['bad_channel_value']
    _, _, real_ids, _ = analysis.interpolate_data(ori, params, "monkey")
    spont_data_flat = np.array(spont_data.flat)
    spont_data_orig = spont_data_flat

    # set reference, controls and test values for given parameters
    interpolated = analysis.interpolate_data(ori, params, "monkey_orientation")
    if params['control'] == 'unif':
        spont_data_real = np.zeros(64)
        for i, id in enumerate(real_ids):
            spont_data_real[i] = spont_data[id[0], id[1]]
        controls = np.random.uniform(0, 180, size=(params["permutations"], 64))
        controls[:, np.where(spont_data_real < 0)] = params['bad_channel_value']
        if params['control_type'] == "orientation":
            reference = remove_bad_channels_OP_map(interpolated[0], spont_data_flat, params)
            test = spont_data_flat
        elif params['control_type'] == "spontaneous":
            reference = remove_bad_channels_OP_map(interpolated[0], spont_data_flat, params)
            test = spont_data_flat
    elif params['control'] == 'perm':
        if params['control_type'] == "orientation":
            controls = permutate_OP_map(ori.reshape(64), size=params["permutations"], empty_val=params['bad_channel_value'])
            reference = spont_data_flat
            test = remove_bad_channels_OP_map(interpolated[0], spont_data_flat, params)
            # test = remove_untunned_channels(interpolated, ori, params, array_n)
        elif params['control_type'] == "spontaneous":
            spont_data_real = np.zeros(64)
            for i, id in enumerate(real_ids):
                spont_data_real[i] = spont_data[id[0], id[1]]
            controls = permutate_OP_map(spont_data_real, size=params["permutations"], empty_val=params['bad_channel_value'])
            reference = remove_bad_channels_OP_map(interpolated[0], spont_data_flat, params)
                        # np.array(remove_untunned_channels(interpolated, ori, params, array_n).flat)
            test = spont_data_flat
        else:
            raise Exception(f"Wrong control type {params['control_type']}.")

    data_controls = {"n": 0}
    if params["control"] == "unif":
        try:
            with open(f"controls/controls_{monkey}{array_n}.pkl", "rb") as f:
                data_controls = pickle.load(f)
            print("Loaded saved controls.")
        except:
            print("Can't find saved controls.")

    if params["control"] == "unif" and data_controls['n'] >= params['permutations']:
        params['permutations'] = data_controls['n']
        distances = data_controls['distances']
        best_solution = data_controls['best_solution']
        best_score = data_controls['best_score']
        save_controls = False
    else:
        distances = []
        best_score = np.inf
        best_solution = None
        if params["control"] == "unif":
            save_controls = params["save_controls"]
        else:
            save_controls = False

        # interpolate, shift and score controls
        for i in range(controls.shape[0]):
            control = controls[i, :].reshape((8, 8))
            control = np.rot90(np.rot90(control.T))  # convert to correct format before interpolation
            interpolated = analysis.interpolate_data(control, params, "monkey_orientation")
            interpolated_control = remove_bad_channels_OP_map(interpolated[0], spont_data_flat, params)
            # interpolated_control = remove_untunned_channels(interpolated, ori, params, array_n)
            shifted_interpolated_control, _ = shift_invert(np.array(interpolated_control.flat),
                                                        vlim=(0, 180), n_sifts=n_shifts,
                                                        empty_val=params['bad_channel_value'])
            s, s_ix = best_distances(shifted_interpolated_control, reference, 180, feature=distance_feature, return_index=True)
            distances.append(s)
            if score(s) < best_score:
                best_solution = shifted_interpolated_control[s_ix, :]
                best_score = score(s)

    # interpolation, shift and score for test
    shifted_interpolated_test, transforms = shift_invert(test, vlim=(0, 180), n_sifts=n_shifts, empty_val=params['bad_channel_value'])
    s, ix = best_distances(shifted_interpolated_test, reference, 180, return_index=True, feature=distance_feature)
    transform = transforms[ix]
    test = shifted_interpolated_test[ix, :]
    test = remove_bad_channels_OP_map(test, spont_data_orig, params)

    # percentiles for each position
    distances = np.array(distances)
    percentiles = np.full_like(s, -1)

    # s = s.reshape((65,65))[real_inds[:,0], real_inds[:,1]]
    # distances = np.array([distances[i, :].reshape((65, 65))[real_inds[:, 0], real_inds[:, 1]] for i in range(5000)])
    # scores = np.array([score(distances[i, :]) for i in range(5000)])
    #
    # percentiles = np.full(64, -1)
    for i in range(len(s)):
        if s[i] != 0:
            percentiles[i] = scipy.stats.percentileofscore(distances[:, i], s[i])
    percentiles = percentiles.reshape(spont_shape)
    # percentiles = np.rot90(percentiles.reshape((8,8)).T, k=2)

    if save_controls:
        helper.ensure_dir_exists("controls")
        with open(f"controls/controls_{monkey}{array_n}.pkl", "wb") as f:
            pickle.dump({"n": params["permutations"], "distances": distances, "best_score": best_score, "best_solution": best_solution}, f)
        print("Saved.")

    results = {"Array_ID": array_n, "method": method, "real_inds": real_inds,
               "best_score": best_score, "best_solution": best_solution,
               "control_scores": [score(s) for s in distances], "test_score": score(s),
               # "control_scores": scores, "test_score": score(s),
               "percentile": scipy.stats.percentileofscore([score(s) for s in distances], score(s)),
               # "percentile": scipy.stats.percentileofscore(scores, score(s)),
               "percentiles": percentiles,
               "test": test.reshape(spont_shape),
               "test_flat": test,
               "reference": reference.reshape(spont_shape),
               "transform": transform}
    with open(f"{params['results_path']}/testing_results_{monkey}{array_n}_{method}_{params['remove_PCA_dims']}.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"Saved: {monkey}{array_n}_{method}: {results['percentile']}")
    plotting.plot_test(params, f"{monkey}{array_n}_{method}")