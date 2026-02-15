import polars as pl
import pandas as pd
from pathlib import Path

DPCLUST_DIR = Path('/hot/user/nmatulionis/software/dpclust/output/deepsomatic_filtered_w_sage_ascat')


def get_dpclust_sample_id(sample_id:str):
    sample_base =sample_id.split('_')[0]
    return sample_base


    
def load_clusters(
    sample_id: str,
) -> pl.DataFrame:
    
    dpclust_sample_id = get_dpclust_sample_id(sample_id)
    sample_dir = DPCLUST_DIR/dpclust_sample_id

    cluster_info_path = sample_dir/f'{dpclust_sample_id}_10000iters_1000burnin_bestClusterInfo.txt'

    cluster_info = pl.read_csv(cluster_info_path, separator="\t")
    total_mutations = cluster_info["no.of.mutations"].sum()

    # Compute mutation fractions
    cluster_info = cluster_info.with_columns(
        (pl.col("no.of.mutations") / total_mutations).alias("frac_mutations"),
        (pl.col("no.of.mutations") / total_mutations).alias("cluster_fraction"),
    )

    cluster_info = cluster_info.rename({
        "cluster.no": "cluster_id",
        "location": "cluster_ccf",
    })

    return cluster_info.select(["cluster_id", "cluster_ccf", "cluster_fraction"])

def get_dpinput_path(dpclust_sample_id):
    in_dir = DPCLUST_DIR /dpclust_sample_id
    dpinput_path = in_dir/f'{dpclust_sample_id}_dpInput.txt'
    return dpinput_path
    
    
def load_snv_copies(dpclust_sample_id):
    dpinput_filepath = get_dpinput_path(dpclust_sample_id)
    
    dpinput_df  = pl.read_csv(dpinput_filepath,separator="\t",null_values=['NA'],has_header=False, skip_rows=1,infer_schema_length=100000)
    dpinput_df.columns = ['index','chr', 'start', 'end', 'WT.count', 'mut.count', 'subclonal.CN', 'nMaj1', 'nMin1', 'frac1', 'nMaj2', 'nMin2', 'frac2', 'phase', 'mutation.copy.number', 'subclonal.fraction', 'no.chrs.bearing.mut']
    dpinput_df = dpinput_df.select(['chr','end','no.chrs.bearing.mut'])
    
    dpinput_df = dpinput_df.rename({'chr':'chromosome','end':'position','no.chrs.bearing.mut':'n_copies'}  )
    dpinput_df = dpinput_df.with_columns(("chr" + pl.col("chromosome")).alias("chromosome").cast(pl.Categorical))
    return dpinput_df

def load_cluster_assignments(sample_id: str, clusters: pl.DataFrame) -> pl.DataFrame:
    dpclust_sample_id = get_dpclust_sample_id(sample_id)

    sample_dir = DPCLUST_DIR/dpclust_sample_id
    cluster_assignment_path = sample_dir/ f"{dpclust_sample_id}_10000iters_1000burnin_mutationClusterLikelihoods.bed"
    
    schema_overrides={"chr": pl.Utf8,'most.likely.cluster':pl.Int16}
    cluster_assignment_df = pl.read_csv(cluster_assignment_path, separator="\t",schema_overrides =schema_overrides,null_values=['NA'])

    # Compute max probability across cluster columns
    cluster_cols = [col for col in cluster_assignment_df.columns if col.startswith("prob.cluster")]
    cluster_assignment_df = cluster_assignment_df.with_columns(
        pl.max_horizontal(cluster_cols).alias("cluster_probability")
    )

    # Select and rename relevant columns
    cluster_assignment_df = cluster_assignment_df.select(
        ["chr", "end", "most.likely.cluster", "cluster_probability"]
    ).rename({
        "chr": "chromosome",
        "end": "position",
        "most.likely.cluster": "cluster_id",
    })

    cluster_assignment_df = cluster_assignment_df.with_columns(("chr" + pl.col("chromosome")).alias("chromosome").cast(pl.Categorical))
    
    # Clean up: drop nulls
    cluster_assignment_df = cluster_assignment_df.drop_nulls()

    # Merge with cluster labels
    cluster_assignment_df = cluster_assignment_df.join(
        clusters,
        on="cluster_id",
        how="inner",
    )

    cluster_assignment_df= cluster_assignment_df.select(["chromosome", "position", "cluster_id"])

    n_copies_df = load_snv_copies(dpclust_sample_id)
    
    cluster_assignment_df = cluster_assignment_df.join(n_copies_df,how='inner',on=['chromosome','position'])
    return cluster_assignment_df
