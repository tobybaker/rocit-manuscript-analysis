import sys
import polars as pl

import cn_loader
import cluster_loader
import phasing_loader
import variant_loader
from pathlib import Path
from rocit.preprocessing import tumor_data_labeller
from rocit.constants import HUMAN_CHROMOSOME_ENUM
def get_short_read_variant_filter_type(sample_id:str):
    if sample_id.startswith('BS'):
        return 'Remove_Germline'
    return 'PASS_Intersection'

def run_short_read_filtering(sample_id,long_read_variants,short_read_variants):
    #doing a simple germline filter
    filter_type = get_short_read_variant_filter_type(sample_id)
    
    if filter_type=='Remove_Germline':
        germline_filter = short_read_variants.filter(pl.col('filter').str.contains('Germline'))
        long_read_variants = long_read_variants.join(germline_filter,how='anti',on=['chromosome','position'])
       
    else:
        short_read_pass = short_read_variants.filter(pl.col('filter') =='PASS').drop('filter')
        long_read_variants = long_read_variants.join(short_read_pass,how='inner',on=['chromosome','position','ref','alt'])
    return long_read_variants

def get_bam_path(sample_id:str):
    in_dir = Path('/hot/user/candrasz/no-kinetic-bams/')
    return in_dir/f'{sample_id}.HT_split.primary_nokinetics.bam'

def get_methylation_dir(sample_id:str):
    base_dir = Path(f'/hot/user/tobybaker/ROCIT_Paper/input_data/cpg_methylation')
    return base_dir/sample_id

def get_normal_id(tumor_id):
    return f'{tumor_id.split("_")[0]}_NL'
def load_dbsnp():
    in_dir = Path('/hot/user/datkinson/dbSNP/')
    filename = 'dbSNP_all_TOPMED_001.vcf.gz'
    in_path = in_dir/filename

    schema = {
    "#CHROM": pl.String,
    "POS": pl.Int64,
    "ID": pl.String,
    "REF": pl.String,
    "ALT": pl.String,
    "QUAL": pl.String,
    "FILTER": pl.String,
    "INFO": pl.String,
    }

    in_df  = (
    pl.scan_csv(
        in_path,
        separator="\t",
        comment_prefix="##",
        schema_overrides=schema
    )
    .select(
        (pl.lit("chr") + pl.col("#CHROM")).alias("chromosome").cast(HUMAN_CHROMOSOME_ENUM),
        pl.col("POS").alias("position"),
        pl.col("REF").alias("ref"),
        pl.col("ALT").alias("alt")
        )
    )
    #check for snps
    in_df = in_df.with_columns(
    is_snp=(
        (pl.col("ref").str.len_bytes() == 1)
        & pl.col("alt")
        .str.split(",")
        .list.eval(pl.element().str.len_bytes() == 1)
        .list.all()
    )
    )
    in_df = in_df.filter(pl.col('is_snp')).drop(['ref','alt','is_snp'])
    return in_df.collect()

if __name__ =='__main__':
    sample_ids = ['053_TU','216_TU','244_TU','264_TU','BS14772_TU','BS15145_TU']
    sample_id = sample_ids[int(sys.argv[1])]
    out_dir = Path('/hot/user/tobybaker/ROCIT_Paper/input_data/labelled_data')
    with pl.StringCache():
        sample_cn = cn_loader.load_cn(sample_id)
        
        clusters = cluster_loader.load_clusters(sample_id)

        if sample_id.startswith('BS'):
            snv_cluster_assignments = None
        else:
            snv_cluster_assignments = cluster_loader.load_cluster_assignments(sample_id,clusters)
        
        haplotags = phasing_loader.load_haplotags(sample_id)
        haploblocks = phasing_loader.load_haploblocks(sample_id)
        
        long_read_variants = variant_loader.load_long_read_variants(sample_id,pass_filter=True)
        short_read_variants = variant_loader.load_short_read_variants(sample_id,pass_filter=False).drop(['tumor_ref_count','tumor_alt_count'])
        dbsnp = load_dbsnp()
        
        
        long_read_variants = run_short_read_filtering(sample_id,long_read_variants,short_read_variants)
        
        long_read_variants = long_read_variants.join(dbsnp,how='anti',on=['chromosome','position'])
        
        sample_bam_path = get_bam_path(sample_id)

        methylation_dir = get_methylation_dir(sample_id)
        pretrain_data = tumor_data_labeller.ROCITSomaticData(sample_id,sample_bam_path,methylation_dir,sample_cn,long_read_variants,haplotags,haploblocks,clusters,snv_cluster_assignments)
        
        read_labels = tumor_data_labeller.make_read_labels(pretrain_data)
        labelled_data = tumor_data_labeller.get_labelled_methylation_data(pretrain_data.sample_methylation_dir,read_labels)
        
        out_path = out_dir/f'{sample_id}_labelled_data.parquet'
        labelled_data.sink_parquet(out_path)

        normal_id = get_normal_id(sample_id)
        normal_methylation_dir = get_methylation_dir(normal_id)

        normal_labelled_data = tumor_data_labeller.get_subsampled_methylation_data(normal_methylation_dir,subsample_rate=0.05)
        normal_out_path = out_dir/f'{normal_id}_labelled_data.parquet'
        normal_labelled_data.sink_parquet(normal_out_path)