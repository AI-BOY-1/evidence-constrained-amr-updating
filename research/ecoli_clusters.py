"""Sequence-screen connected components, with shared BioSamples linked as well."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv,json
from pathlib import Path
import numpy as np

ROOT = ncs_paths.data_path('experiments/ecoli_bvbrc')

def cohort(threshold=.99):
    with (ROOT/'ecoli_assemblies_jaccard.csv').open(encoding='utf-8') as f:
        rows = list(csv.reader(f))
    names = [Path(p).stem for p in rows[0]]
    matrix = np.asarray(rows[1:], dtype=float)
    assert matrix.shape == (len(names), len(names))
    assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 1)
    parent = {a:a for a in names}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a=parent[a]
        return a
    def union(a,b):
        x,y=find(a),find(b)
        if x!=y: parent[max(x,y)]=min(x,y)
    for i,a in enumerate(names):
        for j in np.flatnonzero(matrix[i,:i]>=threshold): union(a,names[j])
    integrity_path=ROOT/'ecoli_fasta_integrity.json'
    if integrity_path.exists():
        integrity=json.loads(integrity_path.read_text(encoding='utf-8'))
        for group in integrity['duplicate_sequence_groups']:
            available=[a for a in group if a in parent]
            for a in available[1:]:union(available[0],a)
    with (ROOT/'ecoli_human_sr_pairs.tsv').open(encoding='utf-8',newline='') as f:
        pairs=list(csv.DictReader(f,delimiter='\t'))
    biosamples={}
    for r in pairs:
        a,bs=r.get('sequence_id') or r['assembly_accession'],r['biosample_accession']
        if a in parent and bs:
            if bs in biosamples: union(a,biosamples[bs])
            biosamples[bs]=a
    return names,matrix,{a:find(a) for a in names},pairs
