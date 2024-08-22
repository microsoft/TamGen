#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Helper tool to call mmseqs2."""

import contextlib
import logging
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Iterable, Union, Sequence, Dict

# Types.
DatasetName = Union[str, Path]
DatasetIndex = int
FastaSequence = str
FastaEntry = Tuple[DatasetName, DatasetIndex, FastaSequence]
ClusterQueryResult = Dict[DatasetName, Sequence[DatasetIndex]]


@contextlib.contextmanager
def timing(msg: str):
    logging.info('Started %s', msg)
    tic = time.time()
    yield
    toc = time.time()
    logging.info('Finished %s in %.3f seconds', msg, toc - tic)


class MMSeqs2:
    def __init__(
            self, *,
            binary_path: Optional[Path] = None,
            c: float = 0.3,
            min_seq_id: float = 0.0,
    ):
        if binary_path is None:
            binary_path = self.find_binary()
        if binary_path is None:
            raise RuntimeError('Must provide AutoDock-smina binary path.')
        self.binary_path = binary_path
        self.c = c
        self.min_seq_id = min_seq_id

    @staticmethod
    def find_binary() -> Optional[Path]:
        """Find mmseqs2 executable."""
        if os.name == 'nt':
            process = subprocess.Popen(['where.exe', 'mmseqs'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            process = subprocess.Popen(['which', 'mmseqs'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        ret_code = process.wait()

        if ret_code:
            return None

        return Path(stdout.decode().splitlines()[0].strip())

    def check_binary(self) -> bool:
        process = subprocess.Popen([str(self.binary_path), '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, _ = process.communicate()
        ret_code = process.wait()
        return not bool(ret_code)

    def cluster(
            self,
            entries: Iterable[FastaEntry], dataset_priority: Sequence[DatasetName] = None,
    ) -> ClusterQueryResult:
        """Cluster the input FASTA sequences and return representative sequence ids.

        Args:
            entries:
            dataset_priority: A sequence of dataset priorities (high priority before low priority).

        Returns:
            cluster query result, dict of representative sequence ids.
        """
        with tempfile.TemporaryDirectory('mmseqs2') as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            db_fn = tmp_dir / 'db.fasta'

            dataset2id = {}
            with open(db_fn, 'w', encoding='utf-8') as f_db:
                for dataset_name, index, sequence in entries:
                    dataset_id = dataset2id.setdefault(dataset_name, len(dataset2id))
                    f_db.write(f'>{dataset_id}_{index}\n{sequence}\n')
            if dataset_priority is not None:
                id2priority = {dataset2id[name]: priority for priority, name in enumerate(dataset_priority)}
            else:
                id2priority = None

            cmd = [
                str(self.binary_path),
                'easy-cluster',
                str(db_fn),
                str(tmp_dir / 'result'),
                str(tmp_dir / 'tmp'),
                '-c', format(self.c, 'g'),
                '--min-seq-id', format(self.min_seq_id, 'g'),
                '-v', '0',
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            with timing(f'MMseqs2 query'):
                stdout, stderr = process.communicate()
                ret_code = process.wait()
            if ret_code:
                raise RuntimeError(f'MMseqs2 failed\nstderr:\n{stderr.decode("utf-8")}\n')

            # Read clusters.
            cluster_fn = tmp_dir / 'result_cluster.tsv'
            # rep_seq_fn = tmp_dir / 'result_rep_seq.fasta'
            # all_seqs_fn = tmp_dir / 'result_all_seqs.fasta'

            clusters = defaultdict(list)
            with open(cluster_fn, 'r', encoding='utf-8') as f_cluster:
                for line in f_cluster:
                    entry1, entry2 = line.strip().split('\t')
                    _id1_s, index1_s = entry1.split('_')
                    _id2_s, index2_s = entry2.split('_')
                    clusters[(int(_id1_s), int(index1_s))].append((int(_id2_s), int(index2_s)))

            # Pick representative sequences.
            id2dataset = {v: k for k, v in dataset2id.items()}
            query_result = {name: [] for name in dataset2id}
            if id2priority is None:
                for cluster_items in clusters.values():
                    dataset_id, index = cluster_items[0]
                    query_result[id2dataset[dataset_id]].append(index)
            else:
                for cluster_items in clusters.values():
                    best_item = min(cluster_items, key=lambda item: id2priority[item[0]])
                    dataset_id, index = best_item
                    query_result[id2dataset[dataset_id]].append(index)
            for indices in query_result.values():
                indices.sort()
            return query_result
