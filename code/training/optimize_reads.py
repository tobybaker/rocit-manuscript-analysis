import sys
import os
import shutil

import polars as pl
import numpy as np

import torch

from rocit.models import ROCITModel

import datahelper

from torch.utils.data import DataLoader

from pathlib import Path
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def smooth_range_penalty(x, penalty_strength=50.0, smoothness=1.0):
    
    # Calculate how far outside [0,1] each element is
    below_zero = torch.nn.functional.relu(-x)  # Positive when x < 0
    above_one = torch.nn.functional.relu(x - 1)  # Positive when x > 1
    
    # Smooth penalty using polynomial (you can also use exponential)
    penalty = penalty_strength * (
        torch.mean(below_zero ** smoothness) + 
        torch.mean(above_one ** smoothness)
    )
    
    return penalty


class MethylationDataStore:
    def __init__(self):
        self.data = {'read_index':[],'chromosome':[],'positions':[],'original_probability':[],'modified_probability':[],'original_methylation':[],'modified_methylation':[]}
        self.percentiles = np.arange(5,100,5)
    
        self.cell_types = self.load_cell_types()
     
     

    def load_cell_types(self):
        base_dir = Path('/hot/user/tobybaker/ROCIT_Paper/input_data')
        cell_type_path = base_dir/'cell_type_average_methylation_atlas.parquet'

        match_col = 'average_methylation_'

        in_df = pl.read_parquet(cell_type_path)

        cell_types = []

        for col in in_df.columns:
            if col.startswith(match_col):
                cell_types.append(col.replace(match_col,''))
        return cell_types
    
    def update(self,chromosome,read_index,original_probability,new_probability,positions,original_methylation,modified_methylation):
        n_positions = original_methylation.shape[0]

        self.data['read_index'].extend([read_index]*n_positions)
        self.data['chromosome'].extend([chromosome]*n_positions)
        self.data['original_probability'].extend([original_probability]*n_positions)
        self.data['modified_probability'].extend([new_probability]*n_positions)
        
        self.data['positions'].extend(list(positions))
        self.data['original_methylation'].extend(list(original_methylation))
        self.data['modified_methylation'].extend(list(modified_methylation))

        
    def get_df(self):
        return pl.DataFrame(self.data)

def load_frozen_model(checkpoint_path,train_data_store):
    tumor_classifier = ROCITModel.load_from_checkpoint(checkpoint_path).model
    
    tumor_classifier.set_embedding_context(train_data_store.embedding_sources)
    tumor_classifier.to(device)

    tumor_classifier.eval()

    for param in tumor_classifier.parameters():
        param.requires_grad = False
    return tumor_classifier

class BatchProcesser:


    MIN_L0_PENALTY =0.0
    L0_WARMUP_STEPS = 975
    NO_L0_STEPS = 25
    MAX_PERTURBING_NOISE =0.1


    MAX_EPSILON = 0.5
    MIN_EPSILON =0.01
    def __init__(self,tumor_classifier,batch,device,l0_penalty):
        self.device = device
        self.tumor_classifier =tumor_classifier
     
        self.l0_penalty = l0_penalty
        self.position = batch['position']
        self.read_position = batch['read_position'].to(device)
        self.attention_mask = batch['attention_mask'].to(device)
        self.methylation_attend = torch.logical_not(self.attention_mask[:,1:])
        self.n_valid_cpgs = self.methylation_attend.sum().item()
        self.methylation = batch['methylation'].to(device)

        self.cell_map_index = batch['cell_map_index'].to(device)
        self.sample_distribution_index = batch['sample_distribution_index'].to(device)

        self.read_index = batch['read_index']
        self.chromosome = batch['chromosome']


    def compute_l0_loss(self,noise,attention,n_steps):
        epsilon = self.MAX_EPSILON - (self.MAX_EPSILON-self.MIN_EPSILON)*min(max(n_steps-self.NO_L0_STEPS,0),self.L0_WARMUP_STEPS)/self.L0_WARMUP_STEPS
        
        noise_squared = torch.pow(noise,2)
        l0_loss = torch.sum(noise_squared/(noise_squared + epsilon**2),dim=1)
        l0_loss = torch.mean(torch.divide(l0_loss,torch.sum(attention,dim=1).float()))
     
        annealed_penalty = self.MIN_L0_PENALTY+ (self.l0_penalty-self.MIN_L0_PENALTY)*min(max(n_steps-self.NO_L0_STEPS,0),self.L0_WARMUP_STEPS)/self.L0_WARMUP_STEPS
        
        return annealed_penalty*l0_loss

    def compute_loss(self,class_out,perturbing_noise,target_signs,n_steps):
        class_loss = -torch.mean(torch.log(torch.sigmoid(class_out*target_signs)))
        l0_penalty = self.compute_l0_loss(perturbing_noise,self.methylation_attend,n_steps)
        
        #L0_penalty = torch.mean(torch.abs(perturbing_noise[self.methylation_attend]))*self.L0_penalty
        clipping_penalty = smooth_range_penalty(self.methylation[self.methylation_attend]+perturbing_noise[self.methylation_attend])
        #print('CLASS',class_loss.item(),'L0',l0_penalty.item(),'Clipping',clipping_penalty.item())
        return class_loss + l0_penalty + clipping_penalty
        
    
    def run_optimisation(self,original_out):
        target_signs = -torch.sign(original_out)
        #skipping the classification token
        perturbing_noise = torch.zeros_like(self.methylation).requires_grad_(True)

        opt = torch.optim.Adam([perturbing_noise], lr=1e-2)
        log_store = []
        for i in range(self.L0_WARMUP_STEPS):
            
            #print(i)
            class_out = self.tumor_classifier(self.methylation+perturbing_noise,self.read_position,self.sample_distribution_index,self.cell_map_index,self.attention_mask) 


            loss =  self.compute_loss(class_out,perturbing_noise,target_signs,i)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            proportion_perturbed = torch.mean((torch.sign(original_out).int()!=torch.sign(class_out).int()).float()).item()
            mean_absolute_noise = (torch.sum(torch.abs(perturbing_noise))/self.n_valid_cpgs).item()
            proportion_big_noise = (torch.sum((torch.abs(perturbing_noise)>0.1).float())/self.n_valid_cpgs).item()
            size_of_big_noise = torch.mean(perturbing_noise[perturbing_noise.abs()>0.1].abs()).item()
            
            below_bounds = (self.methylation+perturbing_noise)<-0.01
            above_bounds = (self.methylation+perturbing_noise)>1.01
            out_of_bounds = torch.logical_and(self.methylation_attend,torch.logical_or(below_bounds,above_bounds))
            out_of_bounds_mean = (torch.sum(out_of_bounds.float())/self.n_valid_cpgs).item()
            
            step_log = {'Step':i,'Loss':loss.item(),'Proportion_Perturbed':proportion_perturbed,'Mean_Absolute_Noise':mean_absolute_noise,'Proportion_Big_Noise':proportion_big_noise,'Size_of_Big_Noise':size_of_big_noise,'Out_of_Bounds':out_of_bounds_mean}
            log_store.append(step_log)
            
           
        perturbing_noise = perturbing_noise.detach()
        perturbing_noise[perturbing_noise.abs()<self.MAX_PERTURBING_NOISE] = 0.0
        return perturbing_noise,pl.DataFrame(log_store)
    
    def get_perturbed_methylation(self):
        
        with torch.no_grad():
            original_out = self.tumor_classifier(self.methylation,self.read_position,self.sample_distribution_index,self.cell_map_index,self.attention_mask)
        
        
        perturbing_noise,perturbing_logs = self.run_optimisation(original_out)
        return original_out,perturbing_noise,perturbing_logs

    def run_update(self,batch_store):

        original_out,perturbing_noise,perturbing_logs = self.get_perturbed_methylation()
        with torch.no_grad():

            final_out = tumor_classifier(torch.clip(self.methylation+perturbing_noise,0.0,1.0),self.read_position,self.sample_distribution_index,self.cell_map_index,self.attention_mask)

        for i in range(self.methylation.shape[0]):
            attention_limit = torch.argmax(batch['attention_mask'][i].long()).item()-1
            if attention_limit ==-1:
                attention_limit = 512

            positions = self.position[i,:attention_limit].cpu().numpy()
            
            read_methylation = self.methylation[i,:attention_limit].cpu().numpy().reshape(-1)
            modified_methylation = read_methylation  + perturbing_noise[i,:attention_limit].detach().cpu().numpy().reshape(-1)
            modified_methylation = np.clip(modified_methylation,0.0,1.0)
             

            original_probability = sigmoid(original_out[i].item())
            new_probability =  sigmoid(final_out[i].item())

            batch_store.update(self.chromosome[i],self.read_index[i],original_probability,new_probability,positions,read_methylation,modified_methylation)
        return perturbing_logs

def get_checkpoint_path(sample_id:str):
    #053_TU_add_normal_False/version_0/checkpoints/best-checkpoint.ckpt
    main_dir = Path('/hot/user/tobybaker/ROCIT_Paper/models/main_models/')
    model_name = f'{sample_id}_add_normal_False'
    model_dir = main_dir/f'{model_name}/version_0/checkpoints'
    return model_dir/'best-checkpoint.ckpt'
if __name__ =='__main__':
    torch.set_float32_matmul_precision('medium')

    device = 'cuda'
    
    sample_id=sys.argv[1]

    checkpoint_path = get_checkpoint_path(sample_id)
    

    out_dir = Path(f'/hot/user/tobybaker/ROCIT_Paper/read_optimizations/{sample_id}')
    
    if out_dir.exists():
        shutil.rmtree(out_dir)

    log_dir = out_dir/'logs'
    log_dir.mkdir(parents=True)

    batch_size =1024
    train_data_store = datahelper.get_sample_train_datasets(sample_id,add_normal=False)
    
    L0_penalties = ['5.0','10.0','15.0']

    
    tumor_classifier = load_frozen_model(checkpoint_path,train_data_store)
    dataloader =  DataLoader(
            train_data_store.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=18,
        )
    for batch_index,batch in enumerate(dataloader):
        
     
        
        for L0_penalty in L0_penalties:
            data_store = MethylationDataStore()
            batch_processor = BatchProcesser(tumor_classifier,batch,device,float(L0_penalty))
            log_df = batch_processor.run_update(data_store)
            out_path = out_dir/f'penalty_{L0_penalty}_batch_index_{batch_index}.parquet'
            data_store.get_df().write_parquet(out_path)

            log_path = log_dir/f'penalty_{L0_penalty}_batch_index_{batch_index}.tsv'
            log_df.write_csv(log_path,separator="\t")
            print('done')
        if batch_index>=9:
            break


                    
        
        
  