import pandas as pd 
import plotting_tools
from pathlib import Path

OUT_DIR = Path('/hot/user/tobybaker/ROCIT_Paper/out_paper/text')
LATEX_TABLE_DIR =Path('/hot/user/tobybaker/ROCIT_Paper/resources/latex_templates')
def get_table(table_data,caption,subwidth=0.8):
    cross_tab = pd.crosstab(table_data['Read Origin'],table_data['Classification']).values

    
    

    subtable_template = load_template(LATEX_TABLE_DIR/'subtabletemplate.txt')
    
    subtable = subtable_template.replace('%SUBWIDTH%',str(subwidth))
    
    subtable = subtable.replace('%NT-NT%',str(cross_tab[0,0]))
    subtable = subtable.replace('%NT-T%',str(cross_tab[0,1]))
    subtable = subtable.replace('%T-NT%',str(cross_tab[1,0]))
    subtable = subtable.replace('%T-T%',str(cross_tab[1,1]))

    subtable = subtable.replace('%CAPTION%',caption)
    return subtable


def load_template(filepath):
    with open(filepath,'r') as out:
        return out.read()


def get_mode_table(overall_data,overall_caption,label,title):
    table_format = load_template(LATEX_TABLE_DIR/'multitabletemplate.txt')
    
    n_modes = overall_data['mode'].nunique()

    mode_count = 0
    subtable_data =''
    for mode,mode_table in overall_data.groupby('mode',observed=True):
        mode_caption = f'{mode} Dataset'
        table_data = get_table(mode_table,mode_caption)
        mode_count +=1
        subtable_data += f'{table_data}\n'
    latex_table = table_format.replace('%SUBTABLES%',subtable_data)
    latex_table = latex_table.replace('%CAPTION%',overall_caption)
    latex_table = latex_table.replace('%LABEL%',label)
    latex_table = latex_table.replace('%TITLE%',title)
    return latex_table

  
def get_latex_document(sample_data,sample_data_summary):

    sample_data = sample_data.to_pandas()
    sample_data = sample_data.rename(columns={'tumor_read':'Read Origin','tumor_assignment':'Classification'})
    sample_data['Read Origin'] = sample_data['Read Origin'].replace({0:'Non-Tumor',1:'Tumor'})
    sample_data['Classification'] = sample_data['Classification'].replace({0:'Non-Tumor',1:'Tumor'})
    main_data =sample_data[(sample_data['add_normal']==False) & (sample_data['out_sample_id']==sample_data['model_sample_id'])]

    latex_data = ''
    for sample_id,sample_data in main_data.groupby('out_sample_id',observed=True):

        mapped_id = plotting_tools.get_sample_mapping()[sample_id]
        caption = f'Confusion matrix for the labeled training, testing and validation datasets for the model trained and evaluated on sample {mapped_id}. A probability threshold of 0.5 was used for tumor read classification.'
        label = f'contigency_{sample_id}'
        sample_latex_table = get_mode_table(sample_data,caption,label,title=mapped_id)
        latex_data += f'{sample_latex_table}\n'
    with open(OUT_DIR/'classification_performance.tex','w') as out_file:
        out_file.write(latex_data)