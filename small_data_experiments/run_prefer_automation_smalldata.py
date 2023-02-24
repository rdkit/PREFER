
import sys
import yaml
import json
import datetime
from os import listdir
from os.path import isfile, join
import warnings
warnings.filterwarnings('ignore')
from prefer.utils.filtering import *
import argparse
from prefer.utils.post_processing_and_optimization_helpers import create_heat_map
from prefer.utils.automation import merge_table_metrics, data_preparation, generate_molecular_representations, run, create_comparison_table



def PREFER_job(data_info):
    # Prepare data
    df = data_preparation(data_info)
    # If time split you need to provide time column name
    temporal_info_column_name = data_info['temporal_info_column_name']
    if (data_info['split_type'] == 'temporal'):
        if(not temporal_info_column_name):
            raise ValueError(f'ERROR: if time split is required then you need to provide the temporal_info_column_name')
    # Extract representations
    representations = generate_molecular_representations(df, split_type = data_info['split_type'],
                                   experiment_name = data_info['experiment_name'] ,
                                   list_of_model_based_representations_paths = data_info['list_of_model_based_representations_paths'])

    # Run PREFER
    bench_list, dir_destination = run(representations, problem_type = data_info['problem_type'], model_instance = data_info['model_instance'])
    # Evaluate results
    merged = merge_table_metrics(bench_list)
    merged.to_csv('merged.csv')
    experiments_dict, tmp_dict = create_comparison_table(merged)
    create_heat_map(experiments_dict, tmp_dict)
    return bench_list, merged, dir_destination


def extract_cddd_path():
    cdddpath = './cddd_representations'
    files = [f for f in listdir(cdddpath) if isfile(join(cdddpath, f))]
    collect_dates = []
    mapping = {}
    for file in files:
        date = file.split('_')[-1]
        date = date.replace('.pkl','')
        date = datetime.datetime.strptime(date, '%Y%m%d-%H%M%S')
        collect_dates.append(date)
        mapping[date] = file

    collect_dates.sort()
    cdddpath = f'{cdddpath}/{mapping[collect_dates[-1]]}'
    print(f'CDDD representations stored as {cdddpath}')
    return cdddpath




def run_PREFER(
    args,
    prefer_args,
):
    """
    This function is used to run automatically PREFER given critical information from a .yalm file and the .pkl cddd representations file.
    The configuration file should contain the following fields:
    - path_to_df
    - experiment_name
    - id_column_name
    - smiles_column_name
    - splitting_strategy
    - problem_type
    - temporal_info_column_name
    - properties_column_name
    """
    path_to_df=args.path_to_df
    experiment_name=args.experiment_name
    id_column_name=args.id_column_name
    smiles_column_name=args.smiles_column_name
    splitting_strategy=args.splitting_strategy
    problem_type= args.problem_type
    temporal_info_column_name=args.temporal_info_column_name
    model_instance = args.model_instance
    model_based_representations = args.model_based_representations # dictionaries fo model_based_representations with corresponding path to the models
    prefer_path = args.prefer_path
    
    print(f'temporal_info_column_name: {temporal_info_column_name}')
    
    try:
        properties_column_name = json.loads(args.properties_column_name[0])

    except Exception:
        properties_column_name_json_format = json.dumps(args.properties_column_name)
        properties_column_name = json.loads(properties_column_name_json_format)

    print('@@@@@@@@@@@@@  Data loaded')
    
    
    data_info = {'path_to_data': path_to_df,
                 'experiment_name': experiment_name,
                 'id_column_name':id_column_name,
                 'model_instance' : model_instance,
                 'problem_type': problem_type,
                 'smiles_column_name':smiles_column_name,
                 'split_type': splitting_strategy,
                 'temporal_info_column_name': temporal_info_column_name,
                 'properties_column_name_list':properties_column_name, 
                 'list_of_model_based_representations_paths': []}
    
    
    if(model_based_representations):
        for model in model_based_representations.keys():
            print(f'Computing representations for model: {model} ...')
            ### here you should continue
            try:
                import os
                import datetime
                from os import listdir
                from os.path import isfile, join
                path_to_model = model_based_representations[model]['path_to_model']
                model_name = model
                conda_env = model_based_representations[model]['conda_env']
                python_path = model_based_representations[model]['submodule_path']
                run_commands = f'source /usr/prog/scicomp/pythonds/conda/etc/profile.d/conda.sh; conda deactivate; conda activate {conda_env}; PYTHONPATH="{python_path}:{prefer_path}:$PYTHONPATH"; export PYTHONPATH; python compute_model_based_representations.py --prefer_args {prefer_args} --path_to_model {path_to_model} --model_name {model_name}'
                print(f'Current subprocess run--> {run_commands}')
                os.system(run_commands)
                
                model_representations_path = f'./{model_name}_representations_{experiment_name}'
                files = [f for f in listdir(model_representations_path) if isfile(join(model_representations_path, f))]
                collect_dates = []
                mapping = {}
                for file in files:
                    date = file.split('_')[-1]
                    date = date.replace('.pkl','')
                    date = datetime.datetime.strptime(date, '%Y%m%d-%H%M%S')
                    collect_dates.append(date)
                    mapping[date] = file

                collect_dates.sort()
                data_info['list_of_model_based_representations_paths'].append(f'{model_representations_path}/{mapping[collect_dates[-1]]}')
                
                
            except Exception as e:
                print(f'Problem while computing {model} representations. In particular: {e}')
                print(f'{model} representations will not be computed')
    else:
        print('Only traditional molecular representations will be computed since no model_based_representations_args have been prepared.')

    print('@@@@@@@@@@@@@  Launch PREFER')
    bench_list, merged, dir_destination = PREFER_job(data_info)
    print('@@@@@@@@@@@@@  PREFER job ended')
    
    
    # save merged
    import os
    import time
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    exp_name = data_info['experiment_name']
    # save merged
    if (not dir_destination.endswith('/')):
        dir_destination = dir_destination+'/' 
    merged.to_csv(f'{dir_destination}merged-{exp_name}_{timestr}.csv')
    
    for bench in bench_list:
        bench.plot_res('./PREFER_results/')
    return





if __name__ == "__main__":
    '''
    '''
    parser = argparse.ArgumentParser(
        description=f"run PREFER",
    )
    parser.add_argument(
        "--prefer_args",
        type=str,
        help="path to the .yaml file where configuration parameters are stored.",
        required=True,
    )

    parser.add_argument(
        "--model_based_representations_args",
        type=str,
        help="path to the .yaml file where configuration parameters for the model_based_representations are stored",
        required=False,
    )

    args = parser.parse_args()
    a_yaml_file = open(args.prefer_args)
    parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)

    args.path_to_df = parsed_yaml_file["path_to_df"]
    args.experiment_name = parsed_yaml_file["experiment_name"]
    args.id_column_name = parsed_yaml_file["id_column_name"]
    args.smiles_column_name = parsed_yaml_file["smiles_column_name"]
    args.properties_column_name = parsed_yaml_file["properties_column_name_list"]
    args.problem_type = parsed_yaml_file["problem_type"]
    args.splitting_strategy = parsed_yaml_file["splitting_strategy"]
    if('model_instance' in parsed_yaml_file):
        args.model_instance = parsed_yaml_file["model_instance"]
    else:
        args.model_instance = None

    if "temporal_info_column_name" in parsed_yaml_file:
        args.temporal_info_column_name = parsed_yaml_file["temporal_info_column_name"]
    else:
        args.temporal_info_column_name = None
        
    if(args.model_based_representations_args):    
        a_yaml_file2 = open(args.model_based_representations_args)
        parsed_yaml_file2 = yaml.load(a_yaml_file2, Loader=yaml.FullLoader)
        args.model_based_representations = parsed_yaml_file2["model_based_representations"]
        args.prefer_path = parsed_yaml_file2["prefer_path"]
    else:
        args.model_based_representations = None
        args.prefer_path = None
    
    run_PREFER(args, args.prefer_args)
    