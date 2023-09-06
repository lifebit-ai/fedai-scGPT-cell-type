import os
from pathlib import Path
from typing import Callable, Dict, Tuple, Union
import yaml

import numpy as np


from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

from scipy.sparse import issparse
import scanpy as sc


from scgpt.preprocess import Preprocessor

from scgpt.tokenizer.gene_tokenizer import GeneVocab


with open('model/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
    
    
def load_raw_cell_type_annotation_data(root_dir: str) -> Tuple[dict, dict]:
    data_dir = Path(root_dir)
    adata = sc.read(data_dir / "c_data.h5ad")
    adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
    adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"
    adata.var.set_index(adata.var["gene_name"], inplace=True)
    adata_test.var.set_index(adata.var["gene_name"], inplace=True)
    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")

    # make the batch category column
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    celltypes = adata.obs["celltype"].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()
    
    adata_test = adata[adata.obs["str_batch"] == "1"]
    adata = adata[adata.obs["str_batch"] == "0"]
    
    return adata, adata_test


def get_processor(n_bins: int, filter_gene_by_counts: bool, data_is_raw: bool):
    preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=filter_gene_by_counts,  # step 1
            filter_cell_by_counts=False,  # step 2
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=data_is_raw,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
    
    return preprocessor


def load_gene_vocab(root_dir: str, genes: list) -> dict:
    # settings for input and preprocessing
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    vocab_file = os.path.join(root_dir, "save/scGPT_human/vocab.json")

    vocab = GeneVocab.from_file(vocab_file)
    
    load_model = os.path.join(root_dir, "save/scGPT_human")
    
    if load_model is None:
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
        
    return vocab


def get_all_counts(adata: sc.AnnData):
    
    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[config["input_style"]]
    
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    
    return all_counts


def get_celltype_labels(adata: sc.AnnData):
    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)
    
    return celltypes_labels


def get_batch_ids(adata: sc.AnnData):
    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)
    
    return batch_ids, num_batch_types

def get_genes(adata: sc.AnnData):
    genes = adata.var["gene_name"].tolist()
    
    return genes

def get_gene_ids(adata: sc.AnnData, vocab: GeneVocab):
    genes = get_genes(adata)
    gene_ids = np.array(vocab(genes), dtype=int)
    
    return gene_ids

def get_processed_data(root_dir):
    data_is_raw = False
    filter_gene_by_counts = False
    
    preprocessor = get_processor(config["n_bins"], data_is_raw, filter_gene_by_counts)
    
    adata, adata_test = load_raw_cell_type_annotation_data(root_dir=root_dir)

    # Preprocess the cell annotation type dataset
    preprocessor(adata, batch_key=None)
    preprocessor(adata_test, batch_key=None)
    
    all_counts = get_all_counts(adata)
    celltypes_labels = get_celltype_labels(adata)
    batch_ids = get_batch_ids(adata)
    genes = get_genes(adata)
    vocab = load_gene_vocab(root_dir, genes)
    gene_ids = get_gene_ids(adata, vocab)
    
    return all_counts, celltypes_labels, batch_ids, gene_ids, vocab