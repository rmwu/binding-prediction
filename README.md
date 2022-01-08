# Binding affinity prediction

You may view sample data files here: https://people.csail.mit.edu/rmwu/static/files/data/

Provided that you have the data, you would run this code via ``train.sh``, in which the variables to change
are marked for you.

As a reference codebase, here is a repo that contains 3 models for the binding affinity regression task:
1) ligand-only (Chemprop) model that ignores the protein
2) a BERT model (Facebook’s ESM), in which the Chemprop ligand representations are incorporated via attention
3) a structure-based model that replaces BERT, based on Wengong's past work

For our datasets, we use two sources, as mentioned.
1) PDBbind (3d protein-ligand complexes with affinity)
2) BindingDB (protein sequence, ligand, affinity).
We select subsets of these data and split them via 30% sequence identity (i.e. train/test proteins share at most 30% sequence id). Note that this is significantly harder than random split / higher sequence id splits.
