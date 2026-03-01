import sys
import polars as pl

import matplotlib.pyplot as plt
import pysam
from supported_vs_unsupported_variant_distribution import load_tumor_predictions
import numpy as np

import matplotlib.colors as mcolors

import matplotlib.cm  as cm
import numpy as np

import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
from rocit.preprocessing import bam_tools
sys.path.insert(0,'../processing')

from make_sample_training_data import get_bam_path
class HandlerVerticalRect(HandlerBase):
    def create_artists(self, legend, orig_handle,
                    xdescent, ydescent, width, height, fontsize, trans):
        # Create a tall, narrow rectangle
        # Make it much taller than it is wide
        rect_width = height   # Narrow width (half of default height)
        rect_height = height * 2  # Tall height (2.5x default height)
        
        # Position the rectangle
        x = xdescent + width * 0.3  # Offset from left
        y = ydescent - height * 0.5  # Center vertically (accounting for extra height)
        
        patch = Rectangle((x, y), rect_width, rect_height,
                        facecolor=orig_handle.get_facecolor(),
                        edgecolor='none',
                        transform=trans)
        return [patch]


def load_read_dfs(vcf_data,sample_id):
    tumor_pacbio_bam_path = get_bam_path(f'{sample_id}_TU')
    tumor_preds = load_tumor_predictions(sample_id).collect()

    read_dfs = []
    for vcf_row in vcf_data.iter_rows(named=True):

        read_df = bam_tools.get_variant_reads(vcf_row,tumor_pacbio_bam_path)
        read_df = read_df.with_columns(pl.col('read_index').cast(pl.Categorical))
        read_df = read_df.join(tumor_preds,how='inner',on=['chromosome','read_index'])
        read_dfs.append(read_df)
    return pl.concat(read_dfs)

if __name__ =="__main__":

    sample_id = '244'

    vcf_data = pl.read_csv('/hot/user/tobybaker/ROCIT_Paper/variant_dir/variant_examples.tsv',separator='\t')
    

    read_dfs = load_read_dfs(vcf_data,sample_id)
    
    
    fig = plt.figure(figsize=(8.9,8/3+0.2))
    gs_top = fig.add_gridspec(2, 1, top=0.88, bottom=0.65, hspace=0.25,left=0.01,right=0.77)
    ax1 = fig.add_subplot(gs_top[0])
    ax2 = fig.add_subplot(gs_top[1])

    # Bottom group (subplots 3 and 4)  
    gs_bottom = fig.add_gridspec(2, 1, top=0.37, bottom=0.14, hspace=0.25,left=0.01,right=0.77)
    ax3 = fig.add_subplot(gs_bottom[0])
    ax4 = fig.add_subplot(gs_bottom[1])

    axs = [ax1, ax2, ax3, ax4]


    colors = ['grey', 'limegreen']
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 0.5, 1.0]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    for plt_index,status in enumerate(['pass','fail']):
        print(read_dfs)
        status_df = read_dfs.filter(pl.col('status')==status)

        status_df = status_df.sort('tumor_probability')

        print(status_df)
        axs[plt_index*2].imshow(status_df['contains_snv'].to_numpy().astype(float).reshape(1,-1),aspect='auto',cmap=cmap,norm=norm,interpolation='none')
        plot = axs[plt_index*2+1].imshow(status_df['tumor_probability'].to_numpy().reshape(1,-1),aspect='auto',cmap='viridis',interpolation='none',vmin=0.0,vmax=1.0)

        axs[plt_index*2].set_xticks([])
        axs[plt_index*2].set_title('Example tumor associated variant' if status =='pass' else 'Example tumor unassociated variant')
        axs[plt_index*2+1].set_xlabel('Reads at variant position')
        for i in range(plt_index*2,plt_index*2+2):
            #axs[i].set_xticks(np.arange(-0.5,len(status_df),1), minor=True)
            #axs[i].grid(which='minor', color='black', linestyle='-', linewidth=1)
            axs[i].tick_params(which='minor', size=0)  # Removes minor tick marks
    for ax in axs:
        ax.set_yticks([])

    # Create colorbar axis - spans the full height
    cbar_ax = fig.add_axes([0.87, 0.17, 0.03, 0.6])  # [left, bottom, width, height]

    # Add colorbar using one of the plot objects (they all have the same scale)
    cbar = fig.colorbar(plot, cax=cbar_ax,label='Tumor Read Probability')
    

    low_color = 'grey'
    high_color = 'limegreen'
    patch1 = mpatches.Patch(color=high_color, label='Read with variant')
    patch2 = mpatches.Patch(color=low_color, label='Read without variant')

    legend = fig.legend(handles=[patch1, patch2],
                        loc='upper right',
                        bbox_to_anchor=(1.0, 1.01),
                        frameon=True,
                    handler_map={mpatches.Patch: HandlerVerticalRect()})
    plt.savefig('/hot/user/tobybaker/ROCIT_Paper/out_paper/plots/snv_calling/variant_tumor_probability_association.png')
    plt.savefig('/hot/user/tobybaker/ROCIT_Paper/out_paper/plots/snv_calling/variant_tumor_probability_association.pdf')


    
   
        
    