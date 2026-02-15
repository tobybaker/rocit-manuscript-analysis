import sys
import shutil
import itertools
import datahelper

from pathlib import Path
from rocit import train,predict,ROCITInferenceStore,TrainingParams
import gc
def clean_and_create_dir(dir_path:Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def run_training_inference(train_result,train_data_store,out_sample_id,experiment_name,train_predictions_dir):
    
    dataset_iter = [('train',train_data_store.train_dataset),('test',train_data_store.test_dataset),('val',train_data_store.val_dataset)]
    for dataset_name,dataset in dataset_iter:
        inference_store = ROCITInferenceStore(dataset,train_data_store.embedding_sources)
        predictions = predict(inference_store,train_result.best_checkpoint_path)

        out_path = train_predictions_dir/f"train_{experiment_name}_out_{out_sample_id}_{dataset_name}_dataset.parquet"
        
        predictions.write_parquet(out_path)



if __name__ =="__main__":
    log_dir = Path('/hot/user/tobybaker/ROCIT_Paper/models/length_models')
    predictions_dir = Path('/hot/user/tobybaker/ROCIT_Paper/predictions/length_predictions')

    param_config = {}
    param_config['Sample_ID'] = ['216_TU','244_TU','264_TU','053_TU','BS14772_TU','BS15145_TU']
    param_config['Read_Length'] = [150,500,1000,2500,5000,7500,10000,12500,15000]
    run_params = datahelper.get_run_params(param_config)
    
    
    run_param = run_params[int(sys.argv[1])]
    sample_predictions_dir = predictions_dir/run_param['Sample_ID']


    sample_predictions_dir = clean_and_create_dir(sample_predictions_dir)
    
    experiment_name = f"{run_param['Sample_ID']}_read_length_{run_param['Read_Length']}"
    
    train_data_store = datahelper.get_sample_train_length_datasets(run_param['Sample_ID'],read_length=run_param['Read_Length'])
    clean_and_create_dir(log_dir/experiment_name)
    t = TrainingParams(max_epochs=1)
    train_result = train(train_data_store,log_dir,experiment_name,training_params=t)
    
    run_training_inference(train_result,train_data_store,run_param['Sample_ID'],experiment_name,sample_predictions_dir)
