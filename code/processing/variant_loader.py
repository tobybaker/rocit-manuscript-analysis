import polars as pl

from pathlib import Path

def get_cancer_type(sample_id):
    if sample_id.startswith('BS'):
        return 'prostate'
    return 'ovarian'

def get_deepsomatic_vcf_path(sample_id):
    sample_cancer_type = get_cancer_type(sample_id)
    
    if sample_cancer_type =='prostate':
        vcf_dir = Path('/hot/user/datkinson/RevioDevRedo/02_SOMATICVARIANTS/PASS')  
    elif sample_cancer_type =='ovarian':
        vcf_dir = Path('/hot/user/datkinson/HGSOC_w_matched_NL/03_SOMATICVARIANTS/PASS/')

    base_id = sample_id.split('_')[0]
    filename = f'{base_id}.deepsomatic_PASS.vcf.gz'
    return vcf_dir/filename

def get_sage_vcf_path(sample_id):
    sample_cancer_type = get_cancer_type(sample_id)
    base_id = sample_id.split('_')[0]
    vcf_dir = Path(f'/hot/user/nmatulionis/short_read/output/{sample_cancer_type}/{base_id}/{base_id}/sage/somatic/')
    filename = f'{base_id}-T.sage.somatic.vcf.gz'
    return vcf_dir/filename

def get_ad_index(df: pl.DataFrame) -> int:
    try:
        first_format = df.select(pl.col("Format")).row(0)[0]
        return first_format.split(':').index('AD')
    except (ValueError, IndexError):
        raise ValueError("Could not find 'AD' field in the 'Format' column.")

def load_vcf(vcf_path: str, mode: str = 'deepsomatic', pass_filter: bool = True) -> pl.DataFrame:

    if mode == 'deepsomatic':
        col_names = ['Chromosome', 'Position', 'ID', 'Ref', 'Alt', 'Qual', 'Filter', 'Info', 'Format', 'Data']
    elif mode == 'sage':
        col_names = ['Chromosome', 'Position', 'ID', 'Ref', 'Alt', 'Qual', 'Filter', 'Info', 'Format', 'Normal_Data', 'Data']
    else:
        raise ValueError(f"Unknown mode: {mode}")

    df = pl.read_csv(
        vcf_path,
        separator='\t',
        has_header=False,
        new_columns=col_names,
        comment_prefix='#'
    )

    # 3. Determine AD index
    ad_col_index = get_ad_index(df)
 
    data_split = pl.col('Data').str.split(':')
    
    genotype_expr = data_split.list.get(0).alias('Genotype')

    ad_part_split = data_split.list.get(ad_col_index).str.split(',')
    
    ref_count_expr = ad_part_split.list.get(0).cast(pl.Int64).alias('Tumor_Ref_Count')
    alt_count_expr = ad_part_split.list.get(1).cast(pl.Int64).alias('Tumor_Alt_Count')

    # 5. Apply transformations
    df = df.with_columns([
        genotype_expr,
        ref_count_expr,
        alt_count_expr
    ])


    df = df.with_columns(
        (pl.col('Tumor_Alt_Count') / (pl.col('Tumor_Alt_Count') + pl.col('Tumor_Ref_Count'))).alias('VAF')
    )

    if pass_filter:
        df = df.filter(pl.col('Filter') == 'PASS')

    if mode == 'deepsomatic':
        df = df.filter(pl.col('Genotype') == '1/1')
 
    df = df.drop(['ID', 'Qual', 'Info', 'Format', 'Data'])
    
    return df



def load_short_read_variants(sample_id):
    bases =['A','C','G','T']
    sage_vcf_path = get_sage_vcf_path(sample_id)
    
    vcf = load_vcf(sage_vcf_path,mode='sage',pass_filter=False)

    vcf = vcf.filter(pl.col('Ref').is_in(bases))
    vcf = vcf.filter(pl.col('Alt').is_in(bases))
    
    return vcf.select(['Chromosome','Position','Ref','Alt','Filter'])


def load_long_read_variants(sample_id):
    bases =['A','C','G','T']
    sage_vcf_path = get_deepsomatic_vcf_path(sample_id)
    
    vcf = load_vcf(sage_vcf_path,mode='deepsomatic')
    vcf = vcf.filter(pl.col('Ref').is_in(bases))
    vcf = vcf.filter(pl.col('Alt').is_in(bases))

    return vcf.select(['Chromosome','Position','Tumor_Ref_Count','Tumor_Alt_Count','VAF'])