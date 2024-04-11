# Paying Attention to SARS-CoV-2 Dialect

This repository contains analysis code, links to data and trained models for the paper "Paying Attention to SARS-CoV-2 Dialect: A
Deep Neural Network Approach to Predicting Novel Mutations" by Magdalyn E. Elkin & Xingquan Zhu. 

## Requirements


* Install dependencies by:

```python
conda create -n dnms python=3.7.9
conda activate dnms
pip install -r requirements.txt
```

## Data

You can download [data here]() and place directory under data/processed so that it has this file path:

```
data
└───processed
    └───ncbi_tree_v1
        │   tree_v1_seq_l2_change_ft.pkl
        │   tree_v1_seq_forward_prob_ft.pkl
        │   tree_v1_seq_attn_ft.pkl
        │   
        └───treebuild
            │   ncov_tree_v1.json
            │   seq_metadata.csv
            │   tree_nodes_v1.pkl
        └───exp_settings
            │   model_folder.txt
            │   mutation_data.pkl
            │   train_seq.pkl
        └───results
            │   pc_tree_v1_ft_l2_forward_mode.csv
            │   pc_tree_v1_ft_l2_forward_mode_summary.csv
            │   ref_ft_l2_forward_mode.csv
            │   ref_ft_l2_forward_mode_summary.csv
```

## Model

You can download our fine tuned [ProtBert model here] and place under directory `logs` so that it has this file path:


```
logs
└───protbert_full_sub_ft_v1
    └───version_2
        └───checkpoints
            └───epoch=3-step=91.ckpt 
```

## Data Files Breakdown

Separate versions of a nextstrain tree can be made, analyzed and recorded with a specific tree version number. 

Tree folders are placed under `data/processed` with the end `v{tree-version}` suffix where `tree-version` indicates 
the tree version. These folders represent the root data folder for experiments.

Root data folders contain: 
* raw calculated ProtBert values for parent and reference sequences
* A `treebuild` folder with raw nextstrain tree, processed tree nodes, and sequence metadata
* A `exp_settings` folder with a text file for the trained model path, mutation data and list of training sequences.
* A `results` folder that contain experimental AUC performance calculations.

Data for analysis and prediction for our work is all found under `data/processed/ncbi_tree_v1`

### Raw Calculations

Calculated Semantic Change, grammaticality and Attention change values for parent sequences and the reference sequence 
are saved under the root data folder as: `tree_v1_seq_l2_change_ft.pkl`, `tree_v1_seq_forward_prob_ft.pkl`, `tree_v1_seq_attn_ft.pkl` respectively. 
These are pickled dictionaries where the key is the sequence and the values are the corresponding calculations for potential mutations.

### Treebuild

Phylogenetic trees in analysis are created with [Nextstrain](https://nextstrain.org/)

The `treebuild` contains:
* the nextstrain auspice json output: `ncov_tree_v1.json`. See [here](https://github.com/nextstrain/augur/blob/master/augur/data/schema-export-v2.json)
for the schema export from Nextstrain.
* The processed tree dictionary: `tree_nodes_v1.pkl` 
  * This is a pickled dictionary with key = NodeName and Value = `PhyloTreeParsers.Node` from `PhyloTreeParsers.py`
* Metadata for sequences used to build the nextstrain tree: `seq_metadata.csv`. 
  * Sequence metadata contains metadata from NCBI and mutation data from [Nextclade CLI](https://docs.nextstrain.org/projects/nextclade/en/stable/index.html) which was run as part of QC prior to building Nextstrain Tree.
  * Sequences were selected from full nucleotide dataset to create Nextstrain as those that represent unique spike sequences. But note that nucleotide sequences are required to create Nextstrain Tree.


### Experiment Settings

Experimental settings are contained in subfolder `exp_settings` which contains:

* Path to finetuned `BioTransformers` ProtBert model saved in `model_folder.txt`
  * `model_folder.txt` lists a simple relative path such as `logs/protbert_full_sub_ft_v1` for the finetuned ProtBert model.
  * Finetuning, Semantic Change and Grammaticality are done with the [Bio-transformers](https://github.com/DeepChainBio/bio-transformers) python wrapper for ProtBert Models
  * Our work uses a pre-trained [ProtBert model](https://github.com/agemagician/ProtTrans)
  * Attention is calculated outside of the Bio-transformers wrapper but with still loading the same finetuned checkpoint as listed in the `model_folder.txt`
* Mutation Data in pickled pandas dataframe: `mutation_data.pkl`
  * Mutation Data is calculated from full `seq_metadata.csv` used to build Nextstrain Tree. A column `test_set_mut` has been created to indicate the mutation was part of our test set. 
  * Mutations in the test set are after a specified cut-off (in our analysis, 1/1/2022). 
  * Mutations in test set also require to be present in child sequences in Nextstrain tree and are applicable for analysis.
    * See section below for Notes on Test Set Mutations for more information.
* Training sequences in pickled list `train_seq.pkl`
  * These are sequences used to train ProtBert and represent sequences from `seq_metadata.csv` that occured more than once time in full nucleotide dataset.
  
#### Notes on Test Set Mutations 

In our analysis, we aim to predict a future mutation by mutating the parent sequence in a phylogenetic tree. 
The sequence of interest is one that has as a child sequence with a novel mutation that happened after a specified cutoff point. 
Mutation summaries are generated from the sequence metadata output from Nextclade. The sequence metadata records mutations *w.r.t* the reference sequence.
At a given point in the phylogenetic tree, the mutation recorded from parent to child may have a different wt mutation.

For example a sequence may have a mutation: K417S, which indicates K at position 417 in reference and S at position 417 in the sequence. 
But in the phylogenetic tree, the node mutation may actually be recorded as N414S. Which indicates the position 417 mutated along the path
from Reference -> Sequence such as K417N -> N417S. In our analysis, we analyze the K417S mutation from parent-child by mutating the parent sequence and attempting to predict mutation N417S, as in the parent sequence, 
N is at position 417. We compare these results against predictions for K417S when mutating the reference sequence. 

Also common is "shifting" positions when comparing mutations in reference against parent-child. This can happen due to deletetions. 
Deletion tokens aren't provided as part of sequences, they are removed to represent the sequence in terms of the natural 20 Amino Acid tokens. 
Thus a mutation such as K811N in the reference may be aligned as K809N in a parent sequence. 

Due to removing deletion tokens if a mutation alignment to the reference is recorded as -809N from parent to child, this mutation isn't applicable to be included in our test set. 
As we require the corresponding mapped wildtype token to be an amino acid in the parent sequence, 
or we're unable to calculate certain language values that's expecting the change from one present token to another token, 
*i.e.* a change from K to N. Thus parent-child mutations such as -809N would be removed from consideration for the parent-child experiment and reference experiment.

### Results

Results subfolder contain full and summary AUC calculations for mutations. Results are separated by reference experiment and parent-child experiments.


# Usage


## Experiments

### FineTuning BioTransformer

To fine tune the biotransformer, you need to provide a log name and path to a pickled list of sequences. Fine tuning will use all GPU available.

```bash
python bin/BioTransFineTune.py --log_name protbert_full_sub_ft_v1 \
    --train_path data/processed/ncbi_tree_v1/exp_settings/train_seq.pkl \
    > biot_ft_log.log 2>&1
```

### Creating processed tree nodes from Nextstrain


To create a processed tree from output of Nextstrain:

```bash
python bin/PhyloTreeParsers.py \
    --nextstrain_path data/processed/ncbi_tree_v1/treebuild/ncov_tree_v1.json \
    --tree_version 1 > phylo_tree.log 2>&1
```

Note our processed tree is availble in the data download.

### Mutation Prediction

Mutation prediction depends on three values calculated from the ProtBert model.

* Grammaticality
* Semantic Change
* Attention Change

For Grammaticality, we used forward mode, which is a single pass through the sequence of interest. 
Semantic Change and Attention Change require calculating semantic embeddings and attention matrices for the sequence of interest and all possible single point amino acid sequences. 
At this time calculations for grammaticality and semantic change are limited to single GPU (if available), 
while attention change can be calculated using multi-GPU. Thus it's recommended to calculate attention change separately. 

Calculations will be saved incrementally for the sequences of interest from the processed tree. 
Sequences of interest are based on those with eligible mutations considering parent-child mutations in the phylogenetic tree.
The Reference sequence is also added to sequences of interest. 

Saved raw calculated grammaticality, semantic change and attention change used in our published results are provided in the data download:

* `data/processed/ncbi_tree_v1/tree_v1_seq_forward_prob_ft.pkl`
  * Probability values calculated with forward mode.
* `data/processed/ncbi_tree_v1/tree_v1_seq_l2_change_ft.pkl`
  * Semantic change calculated with l2 norm
* `data/processed/ncbi_tree_v1/tree_v1_seq_attn_ft.pkl`
  * Attention Change

Each file is a pickled dictionary with sequences as keys. Sequences are mapped to a hash string `parent_hash` listed
in the results output. Running `ParentChildMutate.py` will look for the previous saved values and continue calculations
on identified parent sequences that are missing from saved files.

#### Grammaticality and Semantic Change

To calculate grammaticality and Semantic Change for parent sequences given a processed tree dictionary, mutation data and a finetuned ProtBert model:

```bash
python bin/ParentChildMutate.py --tree_version 1 \
    --seq_values_only --include_change --change_batched \
    --l2_norm --combine --finetuned \
    --data_folder data/processed/ncbi_tree_v1 \
    > pc_mut.log 2>&1
```

#### Attention Change

```bash
python bin/ParentChildMutate.py --tree_version 1 \
    --seq_values_only --include_attn \
    --data_folder data/processed/ncbi_tree_v1 \
    > pc_mut_attn.log 2>&1
```

### Mutation Prediction

After all sequences calculations are saved, they can all be combined and processed for results.
Downloading our previous saved values will also skip calculation steps.

You can also provide alpha, beta and gamma weighting values for the DNMS calculation.
In our experiments, we used alpha=1.5, beta=3.0 and gamma=1.0

```bash
python bin/ParentChildMutate.py --tree_version 1 \
    --include_change --include_attn --l2_norm \
    --combine --finetuned --csv \
    --data_folder data/processed/ncbi_tree_v1 \
    --alpha 1.5 --beta 3.0 --gamma 1.0 \
    > pc_mut_results.log 2>&1
```

### Reference Experiment

Running the command to calculate language model values for the parent sequences will also include the reference sequence. 
Althought the reference sequence is excluded from results when calling `ParentCHildMutate.py`

To only calculate language model values for the reference sequence, or to calculate results for the reference experiment:

```bash
python bin/RefSeqMutate.py --tree_version 1 \
    --include_change --include_attn --l2_norm \
    --combine --finetuned --csv \
    --data_folder data/processed/ncbi_tree_v1 \
    --alpha 1.5 --beta 3.0 --gamma 1.0 \
    > ref_results.log 2>&1
```

Simiarly as stated above, it may be beneficial to calculate values separately at first.







