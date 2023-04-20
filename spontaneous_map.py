"""
Spontaneous maps
This script is used to calculate spontaneous maps from correlation maps using PCA.

Arguments:
    1) monkey_{monkey}{arraynr}_{method}_spont
    2) scan id if running parameter search
    3) number of principal components to running parameter search

Authors: Karolína Korvasová
"""
import helper
import analysis
import yaml
import sys

def main():

    with open('params_paths.yml') as p_file:
        params = yaml.safe_load(p_file)
    with open('params_analysis.yml') as p_file:
        params.update(yaml.safe_load(p_file))

    tag = sys.argv[1]
    if ('scan' in tag) and (len(sys.argv) > 2):
            id = sys.argv[2]
            pca = int(sys.argv[3])
            import pickle
            with open('parameter_space.pkl', "rb") as parspfile:
                pars_update = pickle.load(parspfile)
            print('Updating parameters.')
            print(pars_update['parameters'][int(id)])
            params.update(pars_update['parameters'][int(id)])
            params['remove_PCA_dims'] = pca

    params.update({
       'results_path': params['head_folder'],
        'figures_path': params['head_folder']
       })


    helper.ensure_dir_exists(params['results_path'])
    helper.ensure_dir_exists(params['figures_path'])

    # pca map
    print('Starting to calculate PCA maps.')
    analysis.PCA_map(params, tag)

if __name__ == "__main__":
    main()
