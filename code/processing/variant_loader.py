import polars as pl

from pathlib import Path

BASES = ['A','C','G','T']
BASE_ENUM =  pl.Enum(['A','C','G','T'])
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
        first_format = df.select(pl.col("format")).row(0)[0]
        return first_format.split(':').index('AD')
    except (ValueError, IndexError):
        raise ValueError("Could not find 'AD' field in the 'format' column.")

def load_vcf(vcf_path: str, mode: str = 'deepsomatic', pass_filter: bool = True) -> pl.DataFrame:

    if mode == 'deepsomatic':
        col_names = ['chromosome', 'position', 'id', 'ref', 'alt', 'qual', 'filter', 'info', 'format', 'data']
    elif mode == 'sage':
        col_names = ['chromosome', 'position', 'id', 'ref', 'alt', 'qual', 'filter', 'info', 'format','normal_data', 'data']
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
 
    data_split = pl.col('data').str.split(':')
    
    genotype_expr = data_split.list.get(0).alias('genotype')

    ad_part_split = data_split.list.get(ad_col_index).str.split(',')
    
    ref_count_expr = ad_part_split.list.get(0).cast(pl.Int64).alias('tumor_ref_count')
    alt_count_expr = ad_part_split.list.get(1).cast(pl.Int64).alias('tumor_alt_count')

    # 5. Apply transformations
    df = df.with_columns([
        genotype_expr,
        ref_count_expr,
        alt_count_expr
    ])


    df = df.with_columns(
        (pl.col('tumor_alt_count') / (pl.col('tumor_alt_count') + pl.col('tumor_ref_count'))).alias('vaf')
    )

    if pass_filter:
        df = df.filter(pl.col('filter') == 'PASS')

    if mode == 'deepsomatic':
        df = df.filter(pl.col('genotype') == '1/1')
 
    df = df.drop(['id', 'qual', 'info', 'format', 'data'])
    
    return df



def load_short_read_variants(sample_id):
    sage_vcf_path = get_sage_vcf_path(sample_id)
    
    vcf = load_vcf(sage_vcf_path,mode='sage',pass_filter=False)

    vcf = vcf.filter(pl.col('ref').is_in(BASES))
    vcf = vcf.filter(pl.col('alt').is_in(BASES))
    vcf = vcf.with_columns(pl.col('chromosome').cast(pl.Categorical),pl.col('ref').cast(BASE_ENUM),pl.col('alt').cast(BASE_ENUM))
    return vcf.select(['chromosome','position','ref','alt','filter'])


def load_long_read_variants(sample_id):
    sage_vcf_path = get_deepsomatic_vcf_path(sample_id)
    
    vcf = load_vcf(sage_vcf_path,mode='deepsomatic')
    vcf = vcf.filter(pl.col('ref').is_in(BASES))
    vcf = vcf.filter(pl.col('alt').is_in(BASES))
    vcf = vcf.with_columns(pl.col('chromosome').cast(pl.Categorical),pl.col('ref').cast(BASE_ENUM),pl.col('alt').cast(BASE_ENUM))
    return vcf.select(['chromosome','position','ref','alt','tumor_ref_count','tumor_alt_count','vaf'])