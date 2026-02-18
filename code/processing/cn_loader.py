import polars as pl
from rocit.constants import HUMAN_CHROMOSOMES,HUMAN_CHROMOSOME_ENUM
from pathlib import Path

ASCAT_DIR = Path('/hot/user/ngarciadutton/rocit_results/ascat_3.2')
def get_ascat_cn_path(sample_id):
    sample_path =  ASCAT_DIR/f'{sample_id}.segments.txt'
    return sample_path

def get_ascat_purity_path(sample_id):
    sample_path =  ASCAT_DIR/f'{sample_id.split("_")[0]}_purity_ploidy.txt'
    return sample_path

def get_sample_sex(sample_id:str):
    if sample_id.startswith('BS'):
        return 'XY'
    return 'XX'

def get_ascat_purity(sample_id):
    purity_path = get_ascat_purity_path(sample_id)
    purity = pl.read_csv(purity_path,separator='\t')['purity'][0]
    return purity

def load_cn_ascat(sample_id):
    cancer_type = 'prostate' if 'BS' in sample_id else 'ovarian'
    ascat_path = get_ascat_cn_path(sample_id)
    
    schema_overrides={"chr": pl.Utf8}
    ascat_df = pl.read_csv(ascat_path,separator="\t",schema_overrides=schema_overrides)
    ascat_df = ascat_df.rename({'chr':'chromosome','startpos':'segment_start','endpos':'segment_end','nMajor':'major_cn','nMinor':'minor_cn'})
    ascat_df = ascat_df.drop('sample')
    
    ascat_df = ascat_df.with_columns(
    ("chr" + pl.col("chromosome").cast(pl.Utf8)).alias("chromosome"),
    (pl.col('major_cn')+pl.col('minor_cn')).alias('total_cn'),
    (pl.col('segment_end')-pl.col('segment_start')).alias('segment_length'),
    pl.lit(get_ascat_purity(sample_id)).alias('purity')
    )
    ascat_df = ascat_df.filter(pl.col('chromosome').is_in(HUMAN_CHROMOSOMES))
    ascat_df = ascat_df.with_columns(pl.col('chromosome').cast(HUMAN_CHROMOSOME_ENUM))

    sample_sex = get_sample_sex(sample_id)
    haploid_total_cn = (
        (pl.col("chromosome") == "chrY") |
        ((pl.col("chromosome") == "chrX") & (pl.lit(sample_sex) == "XY"))
    )

    ascat_df = ascat_df.with_columns(
        pl.when(haploid_total_cn).then(1).otherwise(2).alias('normal_total_cn')
    )
    ascat_df = ascat_df.with_columns(
        (pl.col('normal_total_cn')-1).alias('normal_minor_cn')
    )
    
    return ascat_df
def load_cn(sample_id,mode='ASCAT'):
    assert mode =='ASCAT'
    cn_df =  load_cn_ascat(sample_id)
    return cn_df
    
