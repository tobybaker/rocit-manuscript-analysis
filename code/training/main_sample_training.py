import sys
import shutil
import torch
import datahelper

from pathlib import Path
from rocit import train,predict,ROCITInferenceStore,TrainingParams


def clean_and_create_dir(dir_path:Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def run_training_inference(train_result,out_sample_ids,experiment_name,train_predictions_dir):
    for out_sample_id in out_sample_ids:
        train_data_store = datahelper.get_sample_train_datasets(out_sample_id,add_normal=False)
        dataset_iter = [('train',train_data_store.train_dataset),('test',train_data_store.test_dataset),('val',train_data_store.val_dataset)]
        for dataset_name,dataset in dataset_iter:
            inference_store = ROCITInferenceStore(dataset,train_data_store.embedding_sources)
            predictions = predict(inference_store,train_result)

            out_path = train_predictions_dir/f"train_{experiment_name}_out_{out_sample_id}_{dataset_name}_dataset.parquet"
            
            predictions.write_parquet(out_path)

def run_full_dataset_inference(train_result,out_sample_id,experiment_name,full_predictions_dir):
    for normal_id in [True,False]:
        if normal_id:
            run_id = datahelper.tumor_to_normal_id(out_sample_id)
        else:
            run_id = out_sample_id
        inference_store = datahelper.get_sample_inference_store(run_id)
        
        predictions = predict(inference_store,train_result)

        out_path = full_predictions_dir/f"train_{experiment_name}_out_{run_id}_all_reads.parquet"
        predictions.write_parquet(out_path)
if __name__ =="__main__":
    torch.set_float32_matmul_precision('medium')
    log_dir = Path('/hot/user/tobybaker/ROCIT_Paper/models/main_models')
    main_predictions_dir = Path('/hot/user/tobybaker/ROCIT_Paper/predictions/main_predictions')

    param_config = {}
    param_config['Sample_ID'] = ['216_TU','244_TU','264_TU','053_TU','BS14772_TU','BS15145_TU']
    param_config['Add_Normal']  = [False,True]


    run_params = datahelper.get_run_params(param_config)
    run_param = run_params[int(sys.argv[1])]
    

    sample_predictions_dir = main_predictions_dir/f"{run_param['Sample_ID']}_add_normal_{run_param['Add_Normal']}_RUN_2"
    train_predictions_dir = clean_and_create_dir(sample_predictions_dir/'train_datasets')
    full_predictions_dir = clean_and_create_dir(sample_predictions_dir/'full_datasets')
    
    experiment_name = f"{run_param['Sample_ID']}_add_normal_{run_param['Add_Normal']}_RUN_2"
    
    train_data_store = datahelper.get_sample_train_datasets(run_param['Sample_ID'],run_param['Add_Normal'])

    clean_and_create_dir(log_dir/experiment_name)
    training_params = TrainingParams(early_stopping_patience=5)
    train_result = train(train_data_store,log_dir,experiment_name,training_params=training_params)
    
    run_full_dataset_inference(train_result,run_param['Sample_ID'],experiment_name,full_predictions_dir)

    run_training_inference(train_result,param_config['Sample_ID'],experiment_name,train_predictions_dir)
