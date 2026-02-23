import os
import polars as pl
import numpy as np
from dataclasses import dataclass


from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from gene_caller import get_gene_data
import sys

import itertools
import shutil
from matplotlib.lines import Line2D

FIRE_COLOR = "#2bff00"

def create_two_color_cmap(bottom_color, top_color):
    cmap = LinearSegmentedColormap.from_list(
        'custom', 
        [(0, bottom_color), (0.5, bottom_color), 
         (0.5, top_color), (1, top_color)]
    )
    cmap.set_bad(color='white')
    cmap.set_under('white')
    cmap.set_over('white')
    return cmap
def get_methylation_cmap():
    cmap = plt.cm.coolwarm.copy()
    cmap.set_under('white')
    cmap.set_over('white')
    cmap.set_bad('#c9c9c9')
    return cmap  
def find_closest_index(arr, target):
    arr = np.asarray(arr)
    # Create a boolean mask that identifies the non-NaN values
    valid_mask = ~np.isnan(arr)
    # Use only valid (non-NaN) values for comparison
    valid_arr = arr[valid_mask]
    if valid_arr.size == 0:
        return None  # Return None if no valid numbers are present
    # Find the index of the closest valid value
    idx = (np.abs(valid_arr - target)).argmin()
    # Return the index within the original array
    return np.where(valid_mask)[0][idx]


def get_gene_plot_coordinates(gene,positions):
    gene_start = min(gene.start,gene.end)
    gene_end = max(gene.start,gene.end)
    pos_strand = gene.strand =='+'
    if pos_strand:
        gene_promoter_start = gene_start-2000
        gene_promoter_end = gene_start
    else:
        gene_promoter_start = gene_end
        gene_promoter_end = gene_end +2000

    print(gene.name,gene.start,gene.end)
    print(gene.name,gene.strand,pos_strand)
    
    gene_plot_start = find_closest_index(positions,gene_start)
    gene_plot_end = find_closest_index(positions,gene_end)
    gene_plot_promoter_start = find_closest_index(positions,gene_promoter_start)
    gene_plot_promoter_end = find_closest_index(positions,gene_promoter_end)
    return GenePlotCoordinates(gene_plot_start,gene_plot_end,gene_plot_promoter_start,gene_plot_promoter_end,pos_strand)

@dataclass
class SignificantRegion:
    """
    Represents a fixed window of CpG positions.

    Attributes:
       
    """

    chromosome: str
    region_start: int
    region_end: int

@dataclass
class GenePlotCoordinates:
    """
    Represents a gene plotting coordinates

    Attributes:
       
    """

    gene_plot_start:int
    gene_plot_end:int
    gene_plot_promoter_start: int
    gene_plot_promoter_end: int
    gene_pos_strand:bool

def get_hierarchical_sort_order(window_array,in_region):
    window_array = window_array[:,in_region.reshape(-1)==1]
    row_means = np.nanmean(window_array,axis=1)
    window_array = np.nan_to_num(window_array,-10000.0)
    # Compute pairwise distances between rows
    row_distances = pdist(window_array)
    
    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(row_distances, method='average')
    
    # Get the leaf order from the dendrogram
    leaf_order = hierarchy.leaves_list(linkage_matrix)

    cor,p = pearsonr(np.arange(row_means.size),row_means[leaf_order])
    if cor <0:
        leaf_order = leaf_order[::-1]
    return leaf_order

def get_partitioned_hierarchical_sort_order(window_array,in_region,sort_col):
    print('partitioning',window_array.shape)
    sort_order = sorted(list(set(list(sort_col))))

    sort_indicies = []
    for sort_col_val in sort_order:
        
        valid_indices = np.arange(sort_col.size)[sort_col==sort_col_val]
        if valid_indices.size ==0:
            continue
        if valid_indices.size==1:
            sort_indicies.append(valid_indices)
        else:
            sort_order_subset = get_hierarchical_sort_order(window_array[valid_indices,:],in_region)
            sort_order_subset = valid_indices[sort_order_subset]
            sort_indicies.append(sort_order_subset)
    return np.concatenate(sort_indicies)
def get_sort_order(window_array,in_region,discrete_data):
    
    n_vals =discrete_data[list(discrete_data.keys())[0]].size
    combined_sort =[]
    for index in range(n_vals):
        in_str = 'val-'
        for data in discrete_data.values():
            in_str += f'-{data[index]}'
        combined_sort.append(in_str)

    combined_sort = np.array(combined_sort)

    return get_partitioned_hierarchical_sort_order(window_array,in_region,combined_sort)



def add_array_spacing(array:np.array,positions:np.array,small_space=3,small_cutoff=250):
    col_index =0
    spaced_array_store =[]
    positions_diff = np.diff(positions)
    for diff in positions_diff:
        spaced_array_store.append(array[:,col_index])
        if diff >= small_cutoff:
            spaced_array_store.extend([np.ones_like((array[:,col_index]))*np.nan]*small_space)
        col_index +=1
    spaced_array_store.append(array[:,-1])
    return np.column_stack(spaced_array_store)

    
def get_xtick_labels(positions:np.array,min_spacing:int = 20):
    xticks = []
    xtick_labels = []
    add_following_nan = False
    spacing = max(min_spacing,positions.size//20)
    for i in range(positions.size):
        if np.isnan(positions[i]):
            add_following_nan = True
            continue
        if len(xticks)==0 or add_following_nan or i - xticks[-1]>=spacing:
            xticks.append(i)
            xtick_labels.append(f'{int(positions[i]-np.nanmin(positions)):,}')
            if add_following_nan:
                add_following_nan = False
    return xticks,xtick_labels

def run_gene_plotting(ax,genes,positions):
    ax.set_ylim(0,len(genes)+1)
    gene_height = 0.6
    for i,gene in enumerate(genes):
        gene_y = 0.5 + i
        gene_coordinates = get_gene_plot_coordinates(gene,positions)

        gene_box = patches.Rectangle((gene_coordinates.gene_plot_start, gene_y-gene_height/2), gene_coordinates.gene_plot_end-gene_coordinates.gene_plot_start, gene_height, linewidth=0, facecolor='#EF8543')
        #promoter_box = patches.Rectangle((gene_coordinates.gene_plot_promoter_start, gene_y-gene_height/2), gene_coordinates.gene_plot_promoter_end-gene_coordinates.gene_plot_promoter_start, gene_height, linewidth=0, facecolor='#1D92AF')
        # Add the box to the plot
        ax.add_patch(gene_box)
        #ax.add_patch(promoter_box)
        
        gene_name = gene.gene_name if gene.gene_name != '' else gene.gene_id
        ax.text((gene_coordinates.gene_plot_start+gene_coordinates.gene_plot_end)/2, gene_y+0.4, gene_name, fontsize=12, color='black')
        gene_width = gene_coordinates.gene_plot_end-gene_coordinates.gene_plot_start
        arrow_left = gene_coordinates.gene_plot_start+gene_width*0.2
        arrow_right = gene_coordinates.gene_plot_start+gene_width*0.8

        arrow_start = arrow_left if gene_coordinates.gene_pos_strand else arrow_right
        arrow_end = arrow_right if gene_coordinates.gene_pos_strand else arrow_left
        ax.annotate(
            "",
            xy=(arrow_end,gene_y),
            xytext=(arrow_start,gene_y),              
            arrowprops=dict(
                arrowstyle="->",         # Style of the arrow
                color="black",             # Color of the arrow
                lw=2                    # Line width
            )
        )

    ax.set_ylim(0,len(genes)+1.5)

def remove_axis_splines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def plot_fire_regions(fire_regions,read_indices,positions,ax):
    min_pos = np.min(positions)
    max_pos = np.max(positions)
    for row in fire_regions.iter_rows(named=True):
        if not row['read_index'] in read_indices:
            continue
        y_index = read_indices.index(row['read_index'])
        fire_pos = np.interp([max(min_pos,row['start']),min(max_pos,row['end'])],positions,np.arange(positions.size))
        ax.plot(fire_pos,[y_index,y_index],color=FIRE_COLOR,lw=3,alpha=0.7)

def plot_array(window_array:np.array,discrete_data,in_region:np.array,positions:np.array,title:str,out_path:str,genes,fire_regions=None,read_indices=None,add_spacing:bool=False):
    
    sort_order = get_sort_order(window_array,in_region,discrete_data)
    
    window_array = window_array[sort_order]
    for col in discrete_data:
        
        discrete_data[col] = discrete_data[col][sort_order].reshape(-1,1)
        
    if read_indices is not None:
        read_indices = [read_indices[i] for i in sort_order]
    in_region = in_region.reshape(1,-1)

    if add_spacing:
        window_array = add_array_spacing(window_array,positions)
        
        in_region = add_array_spacing(in_region,positions)
        positions = add_array_spacing(positions.reshape(1,-1),positions).reshape(-1)


    region_cmap = create_two_color_cmap('#FFF','#fff129')

    scale_factor = np.sqrt(60.0/(window_array.shape[0]*window_array.shape[1]))
    print('scale factor',scale_factor)
    scale_factor = scale_factor*0.75
    # Create the plot
    n_cols = len(discrete_data)+1
    methylation_col = len(discrete_data)
    width_ratios = [1]*(n_cols-1) + [30]
    fig, axs = plt.subplots(3,n_cols,figsize=(scale_factor*window_array.shape[1],scale_factor*window_array.shape[0]),gridspec_kw={'width_ratios': width_ratios,'height_ratios': [1+6*len(genes),30, 1]},constrained_layout=True)

   
    discrete_lims = {col:(np.nanmin(val-1e-9),np.nanmax(val+1e-9)) for col,val in discrete_data.items()}
    discrete_cmap = {'sample_id':create_two_color_cmap('#383dcf','#0fdb24'),'tumor_read':create_two_color_cmap('#757575','#f731dd'),'haplotag':create_two_color_cmap('#f5be3d','#7b13ab')}
    for col in discrete_data:
        if not col in discrete_cmap:
            discrete_cmap[col] = 'viridis'
    for discrete_count, (col,vals) in enumerate(discrete_data.items()):
        
        axs[1,discrete_count].imshow(vals, cmap=discrete_cmap[col],interpolation='none',vmin=discrete_lims[col][0],vmax=discrete_lims[col][1])
        
        remove_axis_splines(axs[1,discrete_count])
        axs[1,discrete_count].set_ylabel(col.replace('_',' ').title())
        axs[1,discrete_count].set_aspect('auto')

        axs[1,discrete_count].set_xticks([])
        axs[1,discrete_count].set_yticks([])
        axs[1,discrete_count].set_ylim([0, window_array.shape[0]])
        if discrete_count >0:
            axs[1,discrete_count].sharey(axs[1,0])
    
    #will fail if there are no sorting cols
    axs[1,methylation_col].sharey(axs[1,methylation_col-1])  
    

    
    im = axs[1,methylation_col].imshow(window_array, cmap=get_methylation_cmap(),interpolation='none',vmin=0,vmax=1)

    im2 = axs[2,methylation_col].imshow(in_region, cmap=region_cmap,interpolation='none')

    axs[0,methylation_col].sharex(axs[2,methylation_col])
    
    
    
    # Add a colorbar
    #cbar = plt.colorbar(im, ax=axs[1], ticks=bounds,shrink=0.95,aspect=20)
    cbar = plt.colorbar(im, ax=axs[1,methylation_col],shrink=0.95,aspect=20)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_label('Methylation Probability')

    
    axs[1,methylation_col].set_xlabel('Relative Position of CpG Site')
    
    axs[1,methylation_col].set_ylabel('Read')
    fig.suptitle(title)

    axs[1,methylation_col].set_aspect('auto')
    axs[2,methylation_col].set_aspect('auto')

    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])

    
    axs[0,methylation_col].set_xticks([])
    axs[0,methylation_col].set_yticks([])
    
    
    axs[1,methylation_col].set_ylim([0, window_array.shape[0]])

    axs[2,methylation_col].set_xticks([])
    axs[2,methylation_col].set_yticks([])
    #axs[2,methylation_col].set_xlabel('Region of Interest')

    for i in range(n_cols-1):
        for j in [0,2]:
            axs[j,i].set_visible(False)
    

    xtick_positions,xtick_labels = get_xtick_labels(positions)
    axs[1,methylation_col].set_xticks(xtick_positions)
    axs[1,methylation_col].set_xticklabels(xtick_labels)
    
    run_gene_plotting(axs[0,methylation_col],genes,positions)

    for ax in axs.flatten():
        remove_axis_splines(ax)
    

    if fire_regions is not None:
        plot_fire_regions(fire_regions,read_indices,positions,axs[1,methylation_col])

    # Add FIRE LEGEND
    custom_handles = [
        Line2D([0], [0], color=FIRE_COLOR, linewidth=2, label='FIRE Element')
    ]

    fig.legend(
        handles=custom_handles,
        loc='upper left',
        bbox_to_anchor=(0.75, 0.95),   # fine-tune these to clear the colorbar
        bbox_transform=fig.transFigure,
        frameon=True,
        fontsize=9,
        handlelength=1.5,
    )

    plt.savefig(out_path)
    plt.close(fig)
    print('plotted',out_path)



def get_window_array(read_data_bin,discrete_cols,min_cpgs_per_read,min_cpgs_per_col=1):
    
    read_data_bin = read_data_bin.select(['chromosome', 'position', 'read_index', 'methylation', 'in_region'] + discrete_cols)
    read_data_bin = read_data_bin.sort(['chromosome','position','read_index'])
  
    pivoted_df = read_data_bin.pivot(
        on='position',
        index=['chromosome', 'read_index'] + discrete_cols,
        values='methylation',
    )

    print(pivoted_df)

    in_region = (
        read_data_bin
        .group_by(['chromosome', 'position'])
        .agg(pl.col('in_region').any())
        .sort('position')
        .get_column('in_region')
        .to_numpy()
        .astype(float)
    )
    
    discrete_data = {}
    

    positions = np.array(pivoted_df.columns[4:]).astype(np.int64)
    
    pivoted_df_array=  pivoted_df.drop(['chromosome', 'read_index', 'haplotag', 'tumor_read']).to_numpy()
    
    valid_sums = np.count_nonzero(~np.isnan(pivoted_df_array),axis=1)

    pivoted_df_array=pivoted_df_array[valid_sums>=min_cpgs_per_read]
    
    for col in discrete_cols:
        discrete_data[col] = pivoted_df[col].to_numpy()[valid_sums>=min_cpgs_per_read]
        #if col is string return a unique mapping
        if discrete_data[col].dtype == 'O':
            unique_strings, int_array = np.unique(discrete_data[col], return_inverse=True)
            discrete_data[col] = int_array

    read_indices = list(pivoted_df['read_index'].to_numpy()[valid_sums>=min_cpgs_per_read])
    valid_cols = np.count_nonzero(~np.isnan(pivoted_df_array),axis=0)
    
    pivoted_df_array = pivoted_df_array[:,valid_cols>=min_cpgs_per_col]
    in_region = in_region[valid_cols>=min_cpgs_per_col]
    positions = positions[valid_cols>=min_cpgs_per_col]

    if pivoted_df_array.shape[0] <5 or pivoted_df_array.shape[1]<5:
        raise ValueError(f'not enough data, plotting array has shape {pivoted_df_array.shape}')
    return pivoted_df_array,discrete_data,in_region,positions,read_indices




def plot_region(methylation_data,discrete_cols,chromosome,region_start,region_end,filepath,fire_regions=None,window_buffer=2000,mode='hg38'):
    gene_data = get_gene_data(mode)
    
    ensembl_chromosome = 'X' if chromosome=='chrX' else int(chromosome.replace('chr',''))
    genes = gene_data.genes_at_locus(contig=ensembl_chromosome, position=region_start-window_buffer, end=region_end+window_buffer)
    gene_names = [gene.gene_name if gene.gene_name != '' else gene.gene_id for gene in genes]
    
    significant_region_data = methylation_data.filter(
        (pl.col('chromosome') == chromosome)
        & (pl.col('position') >= region_start - window_buffer)
        & (pl.col('position') <= region_end + window_buffer)
    )

    significant_region_data = significant_region_data.with_columns(
        ((pl.col('position') >= region_start) & (pl.col('position') <= region_end))
        .alias('in_region')
    )

    print(significant_region_data)
    
    
    try:
    
        window_array,discrete_data,in_region,positions,read_indices = get_window_array(significant_region_data,discrete_cols,min_cpgs_per_read=40,min_cpgs_per_col=10)
    except ValueError as e:
        print(e)
        return None
    print(window_array)
    region_title = f'{chromosome} - {np.round(region_start-window_buffer,-2):,}-{np.round(region_end+window_buffer,-2):,}'

    title = f'{region_title}'
    #filename = f'{chromosome}-{np.round(region_start-window_buffer,-2)}-{np.round(region_end+window_buffer,-2)}-{"-".join(gene_names)}.png'

    plot_array(window_array,discrete_data,in_region,positions,title,filepath,genes,fire_regions,read_indices)
        


if __name__ =='__main__':

 
    pass
