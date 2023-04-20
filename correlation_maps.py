"""
Correlation maps
This script is used to calculate correlation maps from signal/spikes data.

Arguments:
    1) monkey_{monkey}{arraynr}_{method}_spont
    2) Monkey "L" or "A"
    3) Array ID 1-16
    4) scan id if running parameter search

Authors: Karolína Korvasová, Matěj Voldřich
"""

import helper
import analysis
import os
import yaml
import sys


def main():
    with open('params_paths.yml') as p_file:
        params = yaml.safe_load(p_file)
    with open('params_analysis.yml') as p_file:
        params.update(yaml.safe_load(p_file))

    if len(sys.argv) > 1:
        tag = sys.argv[1]
    else:
        tag = f'blind_MUA_all'
    print(f'Analyzing {tag}.')

    ext = 'nix'
    keywords = [ext]

    if len(sys.argv) > 2:
        monkey = sys.argv[2]
        arrayid = int(sys.argv[3])
    else:
        print('Specify monkey and arrayid.')
        exit()

    if "spont" in tag:
        exptype = 'Spontaneous'
    elif "evoked" in tag:
        exptype = 'Evoked'
    else:
        print('Experiment type unkwnon.')
        exit()
    # tag = tag+f'_{monkey}{arrayid}'
    if 'MUA' in tag and 'MUAe' not in tag:
        spikes_folder = '{}/spikes_tf{}_no_synchrofacts/{}_{}{}'.format(
            params['data_folder_monkey'], int(10 * params['thr_factor']), exptype, monkey, arrayid)
    elif 'nLFP' in tag:
        spikes_folder = '{}/nLFP_spikes_tf{}/{}_{}{}'.format(
            params['data_folder_monkey'], int(10 * params['thr_factor']), exptype, monkey, arrayid)
    elif 'SUA' in tag:
        spikes_folder = '{}/SUA_spikes/{}_{}{}'.format(
            params['data_folder_monkey'], exptype, monkey, arrayid)
    else:
        spikes_folder = f'/projects/Roelfsema_data/{exptype}_{monkey}{arrayid}'

    # if running a parameter search
    if len(sys.argv) > 4:
        id = sys.argv[4]
        import pickle
        with open('parameter_space.pkl', "rb") as parspfile:
            pars_update = pickle.load(parspfile)
        print('Updating parameters.')
        print(pars_update['parameters'][int(id)])
        params.update(pars_update['parameters'][int(id)])

    if 'MUAe' in tag and 'monkey' in tag:
        paths = helper.find_MUAe_recordings(monkey, arrayid, params)
    elif 'LFP' in tag and 'monkey' in tag:
        paths = helper.find_LFP_recordings(monkey, arrayid, params)
    else:
        paths, filenames = helper.find_recordings(spikes_folder, keywords)  # ['nev'])

    params.update({
        'results_path': params['head_folder']
    })
    helper.ensure_dir_exists(params['results_path'])

    import shutil
    shutil.copy('params_analysis.yml', '{}/params_{}.yml'.format(params['results_path'], tag))

    # correlation and som analysis from pooled frames
    frames_path = os.path.join(params['results_path'],
                               'frames_{}_tf{}.pkl'.format(tag, params['thr_factor']))
    analysis.calculate_correlation_maps_pool(paths, params, tag=tag,
                                             frames_path=frames_path)


if __name__ == "__main__":
    main()
