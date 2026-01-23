import sys
import shutil
import torch
import datahelper
import polars
import pytorch_lightning as pl
from pathlib import Path
from rocit import ROCITInferenceStore,TrainingParams,ROCITTrainResult

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


from rocit.models import ROCITModel,ROCITClassifier
from rocit.data import ROCITDataModule,ReadDataset,EmbeddingStore

class ROCITClassifierCustomInput(ROCITClassifier):

    SCALE_CONSTANT:float = 0.05
    def __init__(self, emb, n_heads, n_blocks,seq_length=511,dropout_rate=0.2,sample_distribution_dim=19,cell_map_dim=84,noise_level=0.02,use_cell_map=False,use_sample_distribution=False):
        super().__init__(emb, n_heads, n_blocks,seq_length,dropout_rate,sample_distribution_dim,cell_map_dim,noise_level)

        self.use_cell_map = use_cell_map
        self.use_sample_distribution = use_sample_distribution
        
        methylation_emb_dim = self.sample_distribution_dim+2 if use_sample_distribution else 2
        self.methylation_embedder  = torch.nn.Sequential(
            torch.nn.Linear(in_features=methylation_emb_dim, out_features=emb),
            torch.nn.Dropout(self.dropout),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=emb, out_features=emb),
            torch.nn.Dropout(self.dropout),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=emb, out_features=emb),
        )

    def forward(self, methylation,read_position,sample_distribution_index,cell_map_index,attention_mask,**kwargs):

        cell_type_methylation_sample = self.cell_map_embedding(cell_map_index)
        
        position_methylation_sample= self.sample_distribution_embedding(sample_distribution_index)
        
        if self.training:
            cell_type_methylation_sample += torch.randn_like(cell_type_methylation_sample)*self.noise_level
            position_methylation_sample += torch.randn_like(position_methylation_sample)*self.noise_level
            methylation += torch.randn_like(methylation)*self.noise_level
        
        if self.use_sample_distribution:
            input_vector = self.methylation_embedder(torch.cat([position_methylation_sample,methylation.unsqueeze(-1),read_position.unsqueeze(-1)],dim=-1))
        else:
            input_vector = self.methylation_embedder(torch.cat([methylation.unsqueeze(-1),read_position.unsqueeze(-1)],dim=-1))
        
        if self.use_cell_map:
            input_vector = input_vector+self.cell_type_embedder(cell_type_methylation_sample)

        pos_emb = self.pos_emb(torch.arange(input_vector.shape[1], device=input_vector.device))[None, :, :].expand_as(input_vector)
        
        input_vector = input_vector + pos_emb*self.SCALE_CONSTANT
     
        class_emb = self.class_vector.view(1,1,-1).expand(input_vector.shape[0],-1,-1)

        input_vector = torch.cat([class_emb,input_vector],dim=1)

        x= self.transformer_encoder(input_vector,src_key_padding_mask=attention_mask.bool())
        out= x[:,0].reshape(x.shape[0],-1)
   
       
        class_probs = self.to_output_probability(out).view(-1)
    
        return class_probs


def train_custom_input(rocit_dataset,log_dir,experiment_name,use_cell_map,use_sample_distribution,training_params=None):
    if training_params is None:
        training_params = TrainingParams()
    torch.set_float32_matmul_precision('high') 
    
    logger = CSVLogger(
    save_dir=log_dir,
    name=experiment_name,  
    )

    early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=training_params.early_stopping_patience,
    mode="min",
    )

    checkpointing = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint",
    )

    data_module = ROCITDataModule(rocit_dataset.train_dataset,rocit_dataset.test_dataset,rocit_dataset.val_dataset,training_params.batch_size,num_workers=7)

    warmup_steps = training_params.warmup_steps

    model = ROCITModel(
    model_dim=training_params.model_dim,
    model_heads=training_params.model_heads,
    model_layers=training_params.model_layers,
    lr=training_params.learning_rate,
    warmup_steps=warmup_steps,
    threshold=training_params.probability_threshold,
    sample_distribution_dim=training_params.sample_distribution_dim,
    cell_map_dim=training_params.cell_map_dim,
    noise_level=training_params.noise_level
    )

    
    model.model = ROCITClassifierCustomInput(training_params.model_dim,training_params.model_heads,training_params.model_layers,sample_distribution_dim=training_params.sample_distribution_dim,cell_map_dim=training_params.cell_map_dim,noise_level=training_params.noise_level,use_cell_map=use_cell_map,use_sample_distribution=use_sample_distribution)
    model.model.set_embedding_context(rocit_dataset.embedding_sources)

    trainer = pl.Trainer(
    max_epochs=training_params.max_epochs,
    accelerator="auto",
    devices="auto",
    gradient_clip_val=training_params.gradient_clip_val,
    callbacks=[early_stopping, checkpointing],
    log_every_n_steps=training_params.n_log_steps,
    logger=logger
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    
    best_checkpoint_path = Path(checkpointing.best_model_path)
    log_dir = Path(logger.log_dir)
    
    return ROCITTrainResult(best_checkpoint_path,log_dir)


def predict_custom_input(inference_datastore,training_result,use_cell_map,use_sample_distribution,inference_batch_size:int=1024):
    
    model = ROCITModel.load_from_checkpoint(training_result.best_checkpoint_path)

    #switching models over 
    state_dict = model.model.state_dict()

    custom_model = ROCITClassifierCustomInput(model.model.emb,model.model.n_heads,model.model.n_blocks,use_cell_map=use_cell_map,use_sample_distribution=use_sample_distribution)
    custom_model.load_state_dict(state_dict)
    model.model = custom_model
    
    model.model.set_embedding_context(inference_datastore.embedding_sources)
    trainer =pl.Trainer(accelerator="auto", devices=1)

    predict_loader =  DataLoader(
            inference_datastore.inference_dataset,
            batch_size=inference_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=7,
        )

    predictions = trainer.predict(model, dataloaders=predict_loader)
    predictions = polars.concat([polars.from_dict(batch) for batch in predictions])
    return predictions


def clean_and_create_dir(dir_path:Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def run_sample_inference(train_result,out_sample_id,experiment_name,train_predictions_dir,use_cell_map,use_sample_distribution):

    train_data_store = datahelper.get_sample_train_datasets(out_sample_id,add_normal=False)
    dataset_iter = [('train',train_data_store.train_dataset),('test',train_data_store.test_dataset),('val',train_data_store.val_dataset)]
    for dataset_name,dataset in dataset_iter:
        out_path = train_predictions_dir/f"{dataset_name}_dataset.parquet"
        print(out_path)
        
        inference_store = ROCITInferenceStore(dataset,train_data_store.embedding_sources)
        predictions = predict_custom_input(inference_store,train_result,use_cell_map,use_sample_distribution)

        
        
        predictions.write_parquet(out_path)

if __name__ =="__main__":
    torch.set_float32_matmul_precision('medium')
    log_dir = Path('/hot/user/tobybaker/ROCIT_Paper/models/custom_input_models')
    main_predictions_dir = Path('/hot/user/tobybaker/ROCIT_Paper/predictions/custom_input_predictions')

    param_config = {}
    param_config['Sample_ID'] = ['216_TU','244_TU','264_TU','053_TU','BS14772_TU','BS15145_TU']
    param_config['Use_Cell_Map']  = [True,False]
    param_config['Use_Sample_Distribution']  = [True,False]

    run_params = datahelper.get_run_params(param_config)

    #the example where both cell maps and sample distributions are done are done in the main training loop
    run_params = [param for param in run_params if not param['Use_Cell_Map'] or not param['Use_Sample_Distribution']]
    
    run_param = run_params[int(sys.argv[1])]


    experiment_name = f"{run_param['Sample_ID']}_use_cell_map_{run_param['Use_Cell_Map']}_use_sample_distribution_{run_param['Use_Sample_Distribution']}"

    sample_predictions_dir = main_predictions_dir/f"{run_param['Sample_ID']}/{experiment_name}"
    sample_predictions_dir = clean_and_create_dir(sample_predictions_dir)

    train_data_store = datahelper.get_sample_train_datasets(run_param['Sample_ID'],add_normal=False)
    
    clean_and_create_dir(log_dir/experiment_name)
1)
    train_result = train_custom_input(train_data_store,log_dir,experiment_name,training_params=None,use_cell_map=run_param['Use_Cell_Map'],use_sample_distribution=run_param['Use_Sample_Distribution'])
    

    run_sample_inference(train_result,run_param['Sample_ID'],experiment_name,sample_predictions_dir,run_param['Use_Cell_Map'], run_param['Use_Sample_Distribution'])
