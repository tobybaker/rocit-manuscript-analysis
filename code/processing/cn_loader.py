import polars as pl

from pathlib import Path


def get_ascat_cn_path(sample_id):
    ascat_dir = Path('/hot/user/ngarciadutton/rocit_results/ascat_3.2')
    sample_path =  ascat_dir/f'{sample_id}.segments.txt'
    return sample_path

def get_sample_sex(sample_id:str):
    if sample_id.startswith('BS'):
        return 'XY'
    return 'XX'

def get_ascat_purity(sample_id):
    sample_purity_lookup = {'BS14772_TU':0.42,'BS15145_TU':0.51,'264_TU':0.83,'216_TU':0.9,'244_TU':0.46,'053_TU':0.57,'192_TU':0.77}
    return sample_purity_lookup[sample_id]

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
    return load_cn_ascat(sample_id)
    