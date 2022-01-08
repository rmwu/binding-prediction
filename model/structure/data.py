import csv
import sys
import os
import numpy as np
import json
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio import SeqIO
from Bio.Data.IUPACData import protein_letters_3to1

K = 20
CDR_LIST = [(27, 38), (56, 65), (105, 117)]
niceprint = np.vectorize(lambda x : "%.3f" % (x,))

def read_chain(chain, cdr=None):
    seq = []
    cdr_seq = []
    coord = {'N': [], "CA": [], "C": [], "O": []}
    for residue in chain.get_residues():
        hetflag, resseq, icode = residue.get_id()
        if hetflag.strip() != '': continue

        name = residue.get_resname()
        name = name[0] + name[1:].lower()
        name = protein_letters_3to1[name]
        seq.append(name)

        for atype in ['N', 'CA', 'C', 'O']:
            atom = residue[atype].get_coord() if atype in residue else np.array(['NaN'] * 3)
            atom = eval(np.array2string(atom, separator=','))
            coord[atype].append(atom)

        if cdr is not None:
            if cdr[0][0] <= resseq <= cdr[0][1]:
                cdr_seq.append('1')
            elif cdr[1][0] <= resseq <= cdr[1][1]:
                cdr_seq.append('2')
            elif cdr[2][0] <= resseq <= cdr[2][1]:
                cdr_seq.append('3')
            else:
                cdr_seq.append('0')

    return ''.join(seq), ''.join(cdr_seq), coord


with open('summary.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in tqdm(reader, total=9754):
        if row['Hchain'] == 'NA' or len(row['model']) == 0:
            continue

        pdb_id = row['pdb']
        model_id = int(row['model'])
        try:
            with open(f'imgt/{pdb_id}.pdb') as pdb_f:
                parser = PDBParser()
                protein = parser.get_structure(pdb_id, pdb_f)
                """
                if row['Hchain'] not in protein[model_id]:
                    continue
                """

                hchain = protein[model_id][row['Hchain']]
                hseq, hcdr, hcoords = read_chain(hchain, cdr=CDR_LIST)

                if len(row['Lchain']) > 0 and row['Lchain'] in protein[model_id]:
                    lchain = protein[model_id][row['Lchain']]
                    lseq, lcdr, lcoords = read_chain(lchain, cdr=CDR_LIST)
                else:
                    lseq = lcdr = lcoords = None

                if len(row['antigen_chain']) > 0 and row['antigen_chain'] in protein[model_id]:
                    achain = protein[model_id][row['antigen_chain']]
                    aseq, _, acoords = read_chain(achain)

                    D = np.array(hcoords['CA'])[None,:,:] - np.array(acoords['CA'])[:,None,:]  # [1, N, 3] - [M, 1, 3]
                    D = np.linalg.norm(D, axis=-1, ord=2)  # [M, N]
                    D = D.min(axis=-1)  # [M]
                    epitope = np.argsort(D)[:K]

                    aseq = ''.join([aseq[i] for i in epitope])
                    for atype in ['N', 'CA', 'C', 'O']:
                        acoords[atype] = [acoords[atype][i] for i in epitope]
                else:
                    aseq = acoords = None

                h = json.dumps({"pdb": pdb_id, "seq": hseq, "cdr": hcdr, "coords": hcoords})
                l = json.dumps({"pdb": pdb_id, "seq": lseq, "cdr": lcdr, "coords": lcoords})
                a = json.dumps({"pdb": pdb_id, "seq": aseq, "coords": acoords})
                print(h)
                print(l)
                print(a)
        except Exception as e:
            print(pdb_id, e, file=sys.stderr)

