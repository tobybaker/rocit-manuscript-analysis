import sys
import polars as pl
import numpy as np
from scipy.stats import mannwhitneyu
import os
from pathlib import Path
import matplotlib.pyplot as plt
#import plotting_tools
from matplotlib.patches import Patch



from supported_vs_unsupported_variant_distribution import load_tumor_predictions
sys.path.insert(0,'../analysis')
import plotting_tools
sys.path.insert(0,'../processing')
from variant_loader import load_short_read_variants,load_long_read_variants

MIN_ALT_COUNT:int = 10

MAX_PACBIO_DIST:int=100


def exclude_proximal(
    a: pl.LazyFrame,
    b: pl.LazyFrame,
    max_dist: int,
    chr_col: str = "chromosome",
    pos_col: str = "position",
) -> pl.LazyFrame:
    """Remove rows from A that are within `max_dist` bp of any row in B on the same chromosome."""
    proximal = (
        a.join_where(
            b.select(chr_col, pos_col),
            pl.col(f"{chr_col}_right") == pl.col(chr_col),
            (pl.col(pos_col) - pl.col(f"{pos_col}_right")).abs() <= max_dist,
            suffix="_right",
        )
        .select(chr_col, pos_col)
        .unique()
    )

    return a.join(proximal, on=[chr_col, pos_col], how="anti")
def format_pvalue(p):
    print(p)
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''
def create_vertical_paired_violins(data_dict,comparison_p_values, sample_ids=None, condition_names=None, 
                                 colors=None, figsize=(10, 3),legend_title=None):
    
    if sample_ids is None:
        sample_ids = list(data_dict.keys())

    if condition_names is None:
        condition_names = list(data_dict[sample_ids[0]].keys())

    if colors is None:
        colors = ['#5496ff','#aa49c4']

    fig, ax = plt.subplots(figsize=figsize)
    
    n_conditions = len(condition_names)
    width = 0.75 / n_conditions
    sample_positions = np.arange(len(sample_ids))
    
    # Calculate offset positions for each condition
    # Use smaller offsets and ensure they don't push violins outside plot area
    offset_range =0.2
    offsets = np.linspace(-offset_range, offset_range, n_conditions)

    for cond_idx, condition in enumerate(condition_names):
        for samp_idx, sample_id in enumerate(sample_ids):
            d = data_dict[sample_id][condition]

            
            vp = ax.violinplot([d], positions=[samp_idx + offsets[cond_idx]], 
                             widths=[width],showextrema=False)
    
            
            vp['bodies'][0].set_facecolor(colors[cond_idx])
            vp['bodies'][0].set_alpha(0.7)
            vp['bodies'][0].set_edgecolor('black')
            
            ax.text(samp_idx-0.08,1.00,format_pvalue(comparison_p_values[sample_id]))
    
    sample_size = {s:f'({d["SAGE Fail"].size:,})  ({d["SAGE Pass"].size:,})' for s,d in data_dict.items()}
    # Customize the plot
    ax.set_xticks(sample_positions)
    ax.set_xticklabels([f'{s}\n{sample_size[s]}' for s in sample_ids])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis limits to ensure all violins are visible
    ax.set_xlim(-0.5, len(sample_ids) - 0.5)
    
    # Add legend
    legend_elements = [Patch(facecolor=colors[i % len(colors)],alpha=0.7, label=condition_names[i])
                      for i in range(len(condition_names))]
    
    ax.legend(handles=legend_elements,bbox_to_anchor=(1.05, 1), loc='upper left',title=legend_title)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax

def load_read_table(sample_id:str,mode:str):
    in_dir = Path('/hot/user/tobybaker/ROCIT_Paper/output/short_read_variants')
    filename = f'{sample_id}_{mode}_reads.parquet'
    in_path = in_dir/filename
    in_df = pl.scan_parquet(in_path)
    in_df = in_df.filter(pl.col('contains_snv'))
    return  in_df


def get_valid_variants(sample_id,read_table):
    count_table = read_table.group_by(['chromosome','position']).agg(pl.col('contains_snv').sum())
    normal_count_table = load_read_table(sample_id,'NL').group_by(['chromosome','position']).agg(pl.col('contains_snv').sum())
    
    count_table = count_table.filter(pl.col('contains_snv')>=MIN_ALT_COUNT)
    normal_count_table = normal_count_table.filter(pl.col('contains_snv')>=0)

    count_table = count_table.join(normal_count_table,how='anti',on=['chromosome','position'])
    

    long_read_variants = load_long_read_variants(sample_id,pass_filter=True).lazy()
    count_table = exclude_proximal(count_table,long_read_variants,max_dist=MAX_PACBIO_DIST)
    return count_table.select(['chromosome','position'])


def get_sample_data(sample_id):

    read_table = load_read_table(sample_id,mode='TU')


    valid_variants = get_valid_variants(sample_id,read_table)
    
    read_table = read_table.join(valid_variants,how='semi',on=['chromosome','position'])
    
    predictions = load_tumor_predictions(sample_id)
    read_table = read_table.join(predictions,how='inner',on=['read_index','chromosome'])
    read_table = read_table.with_columns((pl.col('tumor_probability')>=0.5).alias('tumor_read'))
    aggregate_data = read_table.group_by(['chromosome','position','SAGE_filter_status']).agg(pl.col('tumor_read').mean().alias('average_tumor_read'))


    return aggregate_data.collect()


def get_variant_summary_table(sample_variant_table):
    variant_summary_table = sample_variant_table[['Chromosome','Position','Ref','Alt','PASS_Variant']].copy()
    return variant_summary_table
def get_all_sample_data():
    sample_ids = ['216','244','264','053','BS14772','BS15145']
    sample_data = {}
    variant_summary_store = []
    for sample_id in sample_ids:

        sample_variant_table = get_sample_data(sample_id)
        sample_variant_table = sample_variant_table.with_columns(pl.lit(sample_id).alias('sample_id'))
        variant_summary_store.append(sample_variant_table)
        
        plot_sample_id = plotting_tools.get_sample_mapping()[sample_id]
        sample_data[plot_sample_id] = {}


        sample_data[plot_sample_id]['SAGE Fail'] = sample_variant_table.filter(pl.col('SAGE_filter_status')=='fail')['average_tumor_read'].to_numpy()
        sample_data[plot_sample_id]['SAGE Pass'] = sample_variant_table.filter(pl.col('SAGE_filter_status')=='pass')['average_tumor_read'].to_numpy()
        

    return sample_data,pl.concat(variant_summary_store)

def get_comparison_p_values(sample_data):
    comparison_p_values = {}
    for sample_id,sample_values in sample_data.items():
        comparison_p_values[sample_id] = mannwhitneyu(sample_values['SAGE Pass'],sample_values['SAGE Fail']).pvalue
    return comparison_p_values


if __name__ =='__main__':
    
    sample_data,variant_table = get_all_sample_data()

    
    
    #sample_data = get_fake_sample_data(col,variant_type)
    comparison_p_values  = get_comparison_p_values(sample_data)
    
    colors = ["#0a8d50","#be5916"]
    fig,ax = create_vertical_paired_violins(sample_data,comparison_p_values,list(sample_data.keys()),colors=colors,legend_title='Variant Status')
    ax.set_ylabel('Test')
    ax.set_xlabel('Sample ID')
    #ax.set_title(f'Variant Type : {variant_type.capitalize()}')
    plt.tight_layout()
    plt.savefig('/hot/user/tobybaker/ROCIT_Paper/out_paper/plots/snv_calling/sage_distributions.png')
    plt.savefig('/hot/user/tobybaker/ROCIT_Paper/out_paper/plots/snv_calling/sage_distributions.pdf')

    
            
            
            


            