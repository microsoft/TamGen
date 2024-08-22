#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Evaluate target2drug model outputs."""

import argparse
import csv
import logging
import os
from collections import defaultdict
from pathlib import Path

from fairseq.molecule_utils import config
from fairseq.molecule_utils.basic.smiles_utils import disable_rdkit_log
from fairseq.molecule_utils.tgt2drug_evaluator import EvaluateCase, Tgt2DrugEvaluator


def _get_args():
    parser = argparse.ArgumentParser(description='Evaluate target2drug model outputs.')
    parser.add_argument('hyp_path', help='Generated hypothesis file path.')
    parser.add_argument('dataset_path', help='Dataset path.')
    parser.add_argument('-o', '--output-dir', default=os.getcwd(),
                        help='Output directory, default to %(default)r, :null means no output dir.')
    parser.add_argument('--subset', default='test',
                        help='Evaluate on which subset, default to %(default)r.')
    parser.add_argument('--size-check', action='store_true',
                        help='Ensure hyp size == ref size (whole subset)')
    parser.add_argument('--csm', '--candidate-stat-method', dest='candidate_stat_method',
                        choices=Tgt2DrugEvaluator.CANDIDATE_STAT_METHODS,
                        default='mean', help='How to statistic property value of all candidates, default is %(default)r')
    parser.add_argument('-d', '--run-docking', type=int, default=0,
                        help='Run docking on how many PDB IDs, negative means all, default to %(default)r.')
    parser.add_argument('--docking-mode', choices=Tgt2DrugEvaluator.DOCKING_MODES, default='pdb',
                        help='Docking mode (use PDB structure file or given structure file), default to %(default)r')
    parser.add_argument('-b', '--bleu', action='store_true',
                        help='Calculate BLEU score')
    parser.add_argument('--smina-bin-path', default=None, help='AutoDock-smina binary path.')
    parser.add_argument('--split-cache-path', type=Path, default=None,
                        help='Split PDB file cache path, default to %(default)s')
    parser.add_argument('-c', '--cache-path', metavar='PATH', type=Path, default=None,
                        help='PDB file cache path, default to %(default)s')
    parser.add_argument('--ccd', '--ccd-cache-path', metavar='PATH', dest='ccd_cache_path', type=Path,
                        default=None, help='CCD file cache path, default to %(default)s')
    parser.add_argument('--drc', '--docking-result-cache-path', dest='docking_result_cache_path', type=Path,
                        default=None, help='Docking result cache path (only useful in docking mode "pdb"), '
                                           'default to None (no caching)')
    parser.add_argument('--top-n', type=int, default=1, help='Calculate top-N candidates')
    parser.add_argument('--random', default=False, action='store_true', help='Calculate random-N candidates')
    parser.add_argument('--no-box-center', action='store_true', default=False,
                        help='Disable box center.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')

    args = parser.parse_args()
    if args.output_dir == ':null':
        args.output_dir = None
    else:
        args.output_dir = Path(args.output_dir)
    if args.smina_bin_path is not None:
        args.smina_bin_path = Path(args.smina_bin_path)
    if args.split_cache_path is None:
        args.split_cache_path = config.split_pdb_cache_path()
    if args.cache_path is None:
        args.cache_path = config.pdb_cache_path()
    if args.ccd_cache_path is None:
        args.ccd_cache_path = config.pdb_ccd_path()
    if args.docking_result_cache_path is not None:
        if args.docking_mode != 'pdb':
            logging.warning(f'Result cache path is not supported in docking mode {args.docking_mode}, diable it.')
            args.docking_result_cache_path = None

    logging_level = logging.DEBUG if args.verbose else None
    logging.basicConfig(format='%(levelname)s:%(filename)s:%(message)s', level=logging_level)

    disable_rdkit_log()

    return args


def main():
    args = _get_args()

    subset_name: str = args.subset
    dataset_path = Path(args.dataset_path)
    hyp_path = Path(args.hyp_path)

    src_path = dataset_path / 'src'
    if not src_path.exists():
        logging.warning(f'Dataset source directory {src_path} not found.')
        return

    # Load hypothesis.
    all_hyp_m1 = defaultdict(list)
    with hyp_path.open('r', encoding='utf-8') as f_hyp:
        for line in f_hyp:
            hyp_index, m1 = line.strip().split()            # Indices in current subset (0~N).
            hyp_index = int(hyp_index[2:])
            all_hyp_m1[hyp_index].append(m1)

    # Load references.
    m1_fn = src_path / f'{subset_name}.m1.orig'
    with m1_fn.open('r', encoding='utf-8') as f_m1:
        ref_m1 = [line.strip() for line in f_m1]

    info_fn = src_path / f'{subset_name}-info.csv'
    ref_pdb_id = []
    box_center_id = []
    with info_fn.open('r', encoding='utf-8') as f_info:
        reader = csv.DictReader(f_info)
        for row in reader:
            ref_pdb_id.append(row['pdb_id'] or None)
            center_x = row.get('center_x', None)
            center_y = row.get('center_y', None)
            center_z = row.get('center_z', None)
            if center_x is not None and center_y is not None and center_z is not None:
                box_center_id.append((center_x, center_y, center_z))
            else:
                box_center_id.append(None)

    if len(ref_pdb_id) != len(all_hyp_m1):
        err_msg = f'Size mismatch, ref={len(ref_pdb_id)} != hyp={len(all_hyp_m1)}'
        if args.size_check:
            raise RuntimeError(err_msg)
        else:
            logging.warning(err_msg)

    data_to_eval = [
        EvaluateCase(
            index=i,
            ref_smiles=ref_m1[i],
            hyp_smiles_list=all_hyp_m1[i],
            pdb_id=ref_pdb_id[i],
            box_center=box_center_id[i],
        )
        for i in all_hyp_m1
    ]

    evaluator = Tgt2DrugEvaluator(
        out_dir=args.output_dir,
        testset_name=args.subset,
        top_n=args.top_n,
        random=args.random,
        run_bleu=args.bleu,
        docking_mode=args.docking_mode, docking_first_n=args.run_docking,
        candidate_stat_method=args.candidate_stat_method,
        disable_box_center=args.no_box_center,
        smina_bin_path=args.smina_bin_path,
        split_cache_path=args.split_cache_path,
        pdb_cache_path=args.cache_path,
        ccd_cache_path=args.ccd_cache_path,
        docking_result_cache_path=args.docking_result_cache_path,
    )
    evaluator.evaluate_all(data_to_eval)


if __name__ == '__main__':
    main()
