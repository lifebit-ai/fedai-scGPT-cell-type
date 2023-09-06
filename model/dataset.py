import os
from pathlib import Path

import random
from math import ceil
from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
import scanpy as sc

import scgpt as scg
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt import SubsetsBatchSampler
from scgpt.tokenizer.gene_tokenizer import GeneVocab


import torch
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, sampler

random.seed(42)

logger = scg.logger

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"

pad_token = "<pad>"
max_seq_len = 3001
n_bins = 51
include_zero_gene = False
batch_size = 32

load_model = "data/save/scGPT_human"

# %% validate settings
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

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
    
def load_gene_vocab(genes: list) -> dict:
    # settings for input and preprocessing
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    vocab_file ="data/save/scGPT_human/vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    
    if load_model is None:
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
        
    return vocab

def get_processed_data(root_dir):
    data_is_raw = False
    filter_gene_by_counts = False
    
    preprocessor = get_processor(n_bins, data_is_raw, filter_gene_by_counts)
    
    adata, adata_test = load_raw_cell_type_annotation_data(root_dir=root_dir)

    # Preprocess the cell annotation type dataset
    preprocessor(adata, batch_key=None)
    preprocessor(adata_test, batch_key=None)
    
    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[input_style]

    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)
    
    genes = adata.var["gene_name"].tolist()
    vocab = load_gene_vocab(genes)
    gene_ids = np.array(vocab(genes), dtype=int)
    
    return all_counts, celltypes_labels, batch_ids, gene_ids, vocab

    
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
    
    def shuffle(self):
        random.shuffle(self.data)



def get_random_sample_from_dataloader(
    dataloader: DataLoader, n: int, batch_size: int = 4
) -> DataLoader:
    """
    Returns a random sample of size n from the dataloader

    Args:
        dataloader: Dataloader from which the sample is to be taken
        n: Size of the sample

    Returns: Random sample of size n from the dataloader
    """
    dataset = dataloader.dataset
    indices = list(range(len(dataset)))
    random_indices = random.sample(indices, n)
    random_sampler = sampler.SubsetRandomSampler(random_indices)
    random_dataloader = DataLoader(
        dataset, sampler=random_sampler, batch_size=batch_size
    )
    return random_dataloader


def split_dataset(dataset, n_parts, ith):
    """
    Splits the dataset into n_parts and returns the ith part.

    Args:
        n_parts (int): Number of parts to split the dataset into.
        ith (int): Index of the part to return.

    Returns:
        CustomDataset: The ith part of the dataset.
    """
    assert n_parts > 0 and ith >= 0 and ith < n_parts, "Invalid split parameters."

    num_samples = len(dataset)
    samples_per_part = ceil(num_samples / n_parts)
    start_index = ith * samples_per_part
    end_index = min((ith + 1) * samples_per_part, num_samples)

    data_split = dataset[start_index:end_index]
    
    print(f"Returning the {ith} split from {start_index} to {end_index} of length {len(data_split)}")

    return data_split


def prepare_data_samples(
    tokenized_data: Dict[str, torch.Tensor],
    celltype_labels: np.ndarray,
    batch_labels: np.ndarray,
    sort_seq_batch=False
) -> Dict[str, torch.Tensor]:
    mask_ratio = 0.15  # Define your mask ratio
    mask_value = -1  # Define your mask value
    pad_value = 0  # Define your pad value
    epoch = 1  # Replace this with the appropriate epoch value

    masked_values = random_mask_value(
        tokenized_data["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    
    input_gene_ids = tokenized_data["genes"]
    input_values = masked_values
    target_values = tokenized_data["values"]
    tensor_batch_labels = torch.from_numpy(batch_labels).long()
    tensor_celltype_labels = torch.from_numpy(celltype_labels).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each training batch
        sort_ids = np.argsort(batch_labels)
        input_gene_ids = input_gene_ids[sort_ids]
        input_values = input_values[sort_ids]
        target_values = target_values[sort_ids]
        tensor_batch_labels = tensor_batch_labels[sort_ids]
        tensor_celltype_labels = tensor_celltype_labels[sort_ids]

    data_pt = {
        "gene_ids": input_gene_ids,
        "values": input_values,
        "target_values": target_values,
        "batch_labels": tensor_batch_labels,
        "celltype_labels": tensor_celltype_labels,
    }

    return data_pt


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    per_seq_batch_sample: bool = False,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

# Create a load_data function that returns trainloader, testloader, and num_examples
def load_data(
    batch_size: int = 32,
    root_dir: str = "data/",
    local_train: bool = False,
    **kwargs
):
    """
    Loads the data and returns trainloader, testloader, and num_examples.

    Args:
        batch_size (int): Batch size for the data loaders
        root_dir (str): Path to the root directory containing the images

    Returns:
        trainloader (torch.utils.data.DataLoader): Data loader for the training set
        testloader (torch.utils.data.DataLoader): Data loader for the test set
        num_examples (dict): Dictionary containing the number of examples in the train and test sets
    """
    all_counts, celltypes_labels, batch_ids, gene_ids, vocab = get_processed_data(root_dir)
    
    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
    ) = train_test_split(
            all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
        )
    
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )
    
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )
    
    # Prepare train and valid data samples
    train_data_pt = prepare_data_samples(tokenized_train, train_celltype_labels, train_batch_labels, sort_seq_batch=True)
    valid_data_pt = prepare_data_samples(tokenized_valid, valid_celltype_labels, valid_batch_labels, sort_seq_batch=True)

    # Create data generators
    # Note that we are only using the test and train sets and not the validation set
    trainloader = prepare_dataloader(data_pt=train_data_pt, batch_size=batch_size, shuffle=True)
    testloader = prepare_dataloader(data_pt=valid_data_pt, batch_size=batch_size, shuffle=False)
        
    num_examples = {
        "trainset": len(trainloader.dataset),
        "testset": len(testloader.dataset),
    }

    return trainloader, testloader, num_examples
