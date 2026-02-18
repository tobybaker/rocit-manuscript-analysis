
import pandas as pd
import polars as pl
import os

import matplotlib.pyplot as plt
import plotting_tools
import latex_page_maker

import numpy as np

from pathlib import Path
import sys
sys.path.insert(0,'../training/')
import datahelper

def get_total_reads(sample_id:str):
    base_dir = Path('/hot/user/tobybaker/ROCIT_Paper/input_data/cpg_methylation')
    sample_dir = base_dir/f'{sample_id}_TU'

    total_reads = 0
    for filepath in sample_dir.glob('*.parquet'):
        in_dir = pl.scan_parquet(filepath)
        total_reads += in_dir.select('read_index').unique().collect().height
    return total_reads
def get_read_count_data(sample_id:str):
    return_tables = {}
    
    mode_relabel = {'train':'Training','test':'Testing','val':'Validation'}

    base_dir =Path('/hot/user/tobybaker/ROCIT_Paper/input_data/labelled_data')

    labelled_reads_path = base_dir/f'{sample_id}_TU_labelled_data.parquet'
    labelled_reads = pl.read_parquet(labelled_reads_path).select(['read_index','chromosome','tumor_read']).unique()
    labelled_counts = labelled_reads.select('read_index').unique().height
    mode_chromosomes = [('train',datahelper.TRAIN_CHROMOSOMES),('test',datahelper.TEST_CHROMOSOMES),('val',datahelper.VAL_CHROMOSOMES)]
    
    for mode,use_chromosomes in mode_chromosomes:
        mode_df = labelled_reads.filter(pl.col('chromosome').is_in(use_chromosomes))
        
        sample_data = {}
        sample_data['Sample ID'] = plotting_tools.get_sample_mapping()[sample_id]
        sample_data[f'{mode_relabel[mode]} Tumor Reads'] = mode_df['tumor_read'].sum()
        sample_data[f'{mode_relabel[mode]} Non-Tumor Reads'] = mode_df.height - mode_df['tumor_read'].sum()
        return_tables[mode_relabel[mode]] = sample_data

    total_reads = get_total_reads(sample_id)
    #total_reads = np.random.randint(800000,1200000)
    remaining_reads = total_reads-labelled_counts
    remaining_reads_data = {'Sample ID':plotting_tools.get_sample_mapping()[sample_id],'Unlabeled Reads':remaining_reads}
    return_tables['Remaining_Reads'] = remaining_reads_data
    return return_tables

def get_latex_df_str(df):
    
    column_format = "|" + "|".join(["c"] * len(df.columns)) + "|"
    latex_str = df.to_latex(
        index=False,
        column_format=column_format,
        escape=False
    )

    # add horizontal rules after every row
    latex_str = latex_str.replace(r"\\", r"\\ \hline")
    for rule in [r"\toprule", r"\midrule", r"\bottomrule"]:
        latex_str = latex_str.replace(rule, "")


    latex_str = latex_str.replace(r"\begin{tabular}{"+column_format+"}", r"\begin{tabular}{"+column_format+r"}\hline", 1)
    latex_str = latex_str.replace(r"\end{tabular}", r"\hline\end{tabular}", 1)
    return latex_str

def add_percentages_to_sample_data(sample_read_count_data):
    total_reads = 0
    for key,sample_data in sample_read_count_data.items():
        for entry,count in sample_data.items():
            if entry =='Sample ID':
                continue
            total_reads += count
    
    percentage_sample_data = {}
    for key,sample_data in sample_read_count_data.items():
        percentage_sample_data[key] = {}
        for entry,count in sample_data.items():
            if entry =='Sample ID':
                percentage_sample_data[key][entry] = count
            else:
                percentage_sample_data[key][entry] = f'{count} ({(count/total_reads)*100:.1f}\\%)'
    return percentage_sample_data
            
def get_sample_summary_counts(sample_id,sample_read_count_data):
    summary_counts = {'Sample_ID':sample_id,'Tumor':0,'Non-Tumor':0,'Unlabeled':0}
    for key,sample_data in sample_read_count_data.items():
        for entry,count in sample_data.items():
            if entry =='Sample ID':
                continue
            elif 'Unlabeled Reads' in entry:
                summary_counts['Unlabeled'] += count
            elif ' Tumor Reads' in entry:
                summary_counts['Tumor'] += count
            elif 'Non-Tumor Reads' in entry:
                summary_counts['Non-Tumor'] += count
     
    return summary_counts

def write_label_summary_text(read_summary_counts):
    tumor_proportion = read_summary_counts['Tumor']/(read_summary_counts['Tumor']+read_summary_counts['Non-Tumor'])
    tumor_proportion = tumor_proportion*100
    label_proportion = (read_summary_counts['Tumor']+read_summary_counts['Non-Tumor'])/(read_summary_counts['Tumor']+read_summary_counts['Non-Tumor']+read_summary_counts['Unlabeled'])
    label_proportion = label_proportion*100
    out_text = f'In total, a mean of {np.mean(label_proportion):.4f}\\% (range {np.min(label_proportion):.4f} - {np.max(label_proportion):.4f}\\%) of reads in each bulk biopsy could be given a label, with a mean of {np.mean(tumor_proportion):.4f}\\% (range {np.min(tumor_proportion):.4f} - {np.max(tumor_proportion):.4f}\\%) of these reads identified as having tumor origin.'
    out_path = '/hot/user/tobybaker/CellTypeClassifier/paper_plots/latex_tables/label_summary_text.txt'
    with open(out_path,'w') as out_file:
        out_file.write(out_text)

def plot_read_label_summary(read_summary_counts):
    read_summary_counts = read_summary_counts.copy()
    
    read_summary_counts[['Tumor','Non-Tumor','Unlabeled']] = read_summary_counts[['Tumor','Non-Tumor','Unlabeled']]/np.sum(read_summary_counts[['Tumor','Non-Tumor','Unlabeled']].values,axis=1).reshape(-1,1)
    
    fig,ax = plt.subplots(1,1,figsize=(3.2,3.5))
    labels = ['Tumor','Non-Tumor','Unlabeled']
    means = [read_summary_counts[label].mean() for label in labels]
    stds = [read_summary_counts[label].std() for label in labels]
    
    bars = ax.bar(range(1,len(means)+1),means,color='#281ee6',yerr=stds,capsize=5)

    for bar in bars:
        height = bar.get_height()
        
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(range(1,len(means)+1))
    ax.set_xticklabels(labels)
    ax.set_title('Read Label')
    ax.set_ylabel('Proportion of reads in sample')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('/hot/user/tobybaker/ROCIT_Paper/out_paper/plots/schematic/read_label_counts.pdf')
    plt.savefig('/hot/user/tobybaker/ROCIT_Paper/out_paper/plots/schematic/read_label_counts.png')

def get_sample_total_reads(sample_read_count_data):
    total_reads = 0
    for mode,mode_data in sample_read_count_data.items():
        for key,val in mode_data.items():
            if 'Reads' in key:
                total_reads += val
    return total_reads
def write_sample_total_reads(sample_total_reads_store):
    out_string = f'Each biopsy contained an an average of {np.mean(sample_total_reads_store):.3E} reads (SD ={np.std(sample_total_reads_store):.3E}).'
    out_dir = '/hot/user/tobybaker/ROCIT_Paper/out_paper/text'
    out_path=  f'{out_dir}/sample_total_reads.txt'
    with open(out_path,'w') as out:
        out.write(out_string)
if __name__ =="__main__":
    sample_ids = ['216','244','264','053','BS14772','BS15145']

    data_store = {'Training':[],'Testing':[],'Validation':[],'Remaining_Reads':[]}
    sample_total_reads_store = []
    out_path = '/hot/user/tobybaker/ROCIT_Paper/out_paper/text/read_label_counts.tex'

    read_summary_counts = []
    for sample_id in sample_ids:
        sample_read_count_data = get_read_count_data(sample_id)
        sample_total_reads = get_sample_total_reads(sample_read_count_data)
        sample_total_reads_store.append(sample_total_reads)
        
        
        sample_summary_counts = get_sample_summary_counts(sample_id,sample_read_count_data)
        read_summary_counts.append(sample_summary_counts)

        sample_percentage_data = add_percentages_to_sample_data(sample_read_count_data)
        for key,sample_data in sample_percentage_data.items():
            data_store[key].append(sample_data)
    
    read_summary_counts = pd.DataFrame(read_summary_counts)
    write_label_summary_text(read_summary_counts)
    write_sample_total_reads(sample_total_reads_store)
    plot_read_label_summary(read_summary_counts)
    
    combined_table_str = ''
    for key,data in data_store.items():
        df = pd.DataFrame(data)
        latex_str = get_latex_df_str(df)
        subtable_template_path = '/hot/user/tobybaker/ROCIT_Paper/resources/latex_templates/subtabletemplate_tabular_insert.txt'
        subtable_str = latex_page_maker.load_template(subtable_template_path)
        subtable_str = subtable_str.replace('%TABULAR%',latex_str)
        combined_table_str +=subtable_str + '\n'
    
    final_layout_template_path = '/hot/user/tobybaker/ROCIT_Paper/resources/latex_templates/multitabletemplatenotitle.txt'
    final_layout_template = latex_page_maker.load_template(final_layout_template_path)
    final_layout_template = final_layout_template.replace('%SUBTABLES%',combined_table_str)
    final_layout_template = final_layout_template.replace('%CAPTION%','Classification counts for ground truth training data for the five samples in our cohort. The remaining reads are the reads for each sample where no ground truth origin label could be obtained. Percentages are expressed as a fraction of the total bulk tumor reads for the sample.')
    final_layout_template = final_layout_template.replace('%LABEL%','ground_truth_counts')
    # save to file
    with open(out_path, "w") as f:
        f.write(final_layout_template)

