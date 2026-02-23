from pyensembl import EnsemblRelease,Genome


def get_gene_data(mode='hg38'):
    if mode =='hg38':
        gtf_path = "/hot/lab_resources/pyensembl_GRCh38/ensembl110/Homo_sapiens.GRCh38.110.gtf.gz"
        fasta_path = "/hot/lab_resources/pyensembl_GRCh38/ensembl110/Homo_sapiens.GRCh38.cdna.all.fa.gz"  # Optional, only needed if you want sequence data
        protein_path = "/hot/lab_resources/pyensembl_GRCh38/ensembl110/Homo_sapiens.GRCh38.pep.all.fa.gz"  # Optional, only needed if you want sequence data
        reference_name = 'GRCh38'
    if mode =='mm10':
        gtf_path = '/hot/lab_resources/gencode.vM10.annotation.gtf'
        reference_name = 'GRCm38'
    data = Genome(
        reference_name=reference_name,
        annotation_name='my_genome_features',
        # annotation_version=None,
        gtf_path_or_url=gtf_path, # Path or URL of GTF file

    )

    # parse GTF and construct database of genomic features
    data.index()

    return data

if __name__ =='__main__':
    #example usage 
    gene_annotator = get_gene_data()
    #note to not use chr, it's 4 not chr4 and X not chrX
    chrom = '4'
    start = 12678987
    end = start+ 50000
    gene_annotator.genes_at_locus(contig=chrom, position=start, end=end)


    gene = gene_annotator.genes_by_name("TP53")[0]

    print("Gene ID:", gene.gene_id)
    print("Chromosome:", gene.contig)
    print("Start:", gene.start)
    print("End:", gene.end)
    print("Strand:", gene.strand)  # 1 for positive strand, -1 for negative