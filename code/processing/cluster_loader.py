import polars as pl

from pathlib import Path

DPCLUST_DIR = Path('/hot/user/nmatulionis/software/dpclust/output/deepsomatic_filtered_w_sage_ascat')


def get_dpclust_sample_id(sample_id:str):
    sample_base =sample_id.split('_')[0]
    return sample_base
    
def load_cluster_labels(
    sample_id: str,
    min_clonal_cluster: float = 0.9,
    max_clonal_cluster: float = 1.1,
    min_clonal_fraction: float = 0.3,
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

    # Define clonal cluster criteria
    is_clonal = (
        pl.col("location").is_between(min_clonal_cluster, max_clonal_cluster)
        & (pl.col("frac_mutations") > min_clonal_fraction)
    )
    is_above_clonal = pl.col("location") >= max_clonal_cluster

    # Assign cluster labels
    cluster_label_expr = (
        pl.when(is_clonal).then(pl.lit("Pass_Clonal"))
        .when(is_above_clonal).then(pl.lit("Fail"))
        .otherwise(pl.lit("Pass"))
        .alias("cluster_label")
    )
    cluster_info = cluster_info.with_columns(cluster_label_expr)

    # Rename and select final columns
    cluster_info = cluster_info.rename({
        "cluster.no": "cluster",
        "location": "cluster_ccf",
    })

    return cluster_info.select(["cluster", "cluster_label", "cluster_ccf", "cluster_fraction"])


def load_cluster_assignments(sample_id: str, cluster_labels: pl.DataFrame) -> pl.DataFrame:
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
        "most.likely.cluster": "cluster",
    })

    cluster_assignment_df = cluster_assignment_df.with_columns(("chr" + pl.col("chromosome")).alias("chromosome"))
    
    # Clean up: drop nulls
    cluster_assignment_df = cluster_assignment_df.drop_nulls()

    # Merge with cluster labels
    cluster_assignment_df = cluster_assignment_df.join(
        cluster_labels,
        on="cluster",
        how="inner",
    )

    return cluster_assignment_df.select(["chromosome", "position", "cluster_label"])

