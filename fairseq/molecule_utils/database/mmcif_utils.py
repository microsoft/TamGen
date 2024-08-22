#! /usr/bin/python
# -*- coding: utf-8 -*-

"""mmCIF file parsing functions."""

from dataclasses import dataclass
from typing import Dict, List, Iterator, Tuple, Optional

from Bio.PDB.Chain import Chain as PdbChain
from Bio.PDB.Residue import Residue as PdbResidue

from .af2_mmcif_parsing import (
    mmcif_loop_to_list, mmcif_loop_to_dict,
    AF2MmcifObject, ResidueAtPosition,
)
from .common_utils import norm_cif_empty_value, is_empty, aa_3to1
from ..exceptions import MultipleChainsError


@dataclass
class UniProtRef:
    pdb_id: str
    align_id: str
    ref_id: str
    uniprot_id: str
    chain_id: str
    seq_begin: int
    seq_end: int
    seq_begin_insertion_code: str
    seq_end_insertion_code: str
    db_seq_begin: int
    db_seq_end: int
    db_seq_begin_insertion_code: str
    db_seq_end_insertion_code: str


def get_uniprot_ref(mmcif_object: AF2MmcifObject) -> Dict[str, UniProtRef]:
    """Get external UniProt ID references (DBREF) of given structure."""
    mmcif_dict = mmcif_object.raw_string

    all_ref_data = mmcif_loop_to_dict('_struct_ref.', '_struct_ref.id', mmcif_dict)
    all_ref_seq_data = mmcif_loop_to_dict('_struct_ref_seq.', '_struct_ref_seq.align_id', mmcif_dict)

    uniprot_ref_data = {}
    for align_id, ref_seq_data in all_ref_seq_data.items():
        ref_id = ref_seq_data['_struct_ref_seq.ref_id']
        ref_data = all_ref_data[ref_id]
        if ref_data['_struct_ref.db_name'] != 'UNP':
            continue
        uniprot_ref_data[align_id] = UniProtRef(
            pdb_id=mmcif_object.file_id,
            align_id=align_id,
            ref_id=ref_id,
            uniprot_id=ref_data['_struct_ref.pdbx_db_accession'].lower(),
            chain_id=ref_seq_data['_struct_ref_seq.pdbx_strand_id'],
            seq_begin=int(ref_seq_data['_struct_ref_seq.seq_align_beg']),
            seq_end=int(ref_seq_data['_struct_ref_seq.seq_align_end']),
            seq_begin_insertion_code=norm_cif_empty_value(
                ref_seq_data['_struct_ref_seq.pdbx_seq_align_beg_ins_code']),
            seq_end_insertion_code=norm_cif_empty_value(
                ref_seq_data['_struct_ref_seq.pdbx_seq_align_end_ins_code']),
            db_seq_begin=int(ref_seq_data['_struct_ref_seq.db_align_beg']),
            db_seq_end=int(ref_seq_data['_struct_ref_seq.db_align_end']),
            db_seq_begin_insertion_code=norm_cif_empty_value(
                ref_seq_data['_struct_ref_seq.pdbx_db_align_beg_ins_code']),
            db_seq_end_insertion_code=norm_cif_empty_value(
                ref_seq_data['_struct_ref_seq.pdbx_db_align_end_ins_code']),
        )
    return uniprot_ref_data


@dataclass(order=True)
class SingleBindingPosition:
    chain_id: str
    res_id: int
    res_name: str
    insertion_code: str

    def to_str(self):
        return f'{self.chain_id}.{self.res_id}'


@dataclass
class BindingSite:
    pdb_id: str
    site_id: str
    binding_positions: List[SingleBindingPosition]

    def binding_positions_to_str(self):
        return ';'.join(f'{pos.chain_id}.{pos.res_id}' for pos in self.binding_positions)

    def pretty_print(self):
        print(f'Binding site {self.pdb_id}:{self.site_id}')
        for bp in self.binding_positions:
            print(f'{bp.res_name} ({aa_3to1(bp.res_name)}) @ {bp.chain_id} {bp.res_id} {bp.insertion_code}')


def get_binding_sites(mmcif_object: AF2MmcifObject, auth_id: bool = False) -> Dict[str, BindingSite]:
    mmcif_dict = mmcif_object.raw_string
    pdb_id = mmcif_object.file_id

    site_structures = mmcif_loop_to_dict('_struct_site_gen.', '_struct_site_gen.id', mmcif_dict)

    site_ids = [item['_struct_site.id'] for item in mmcif_loop_to_list('_struct_site.id', mmcif_dict)]
    result = {site_id: BindingSite(pdb_id=pdb_id, site_id=site_id, binding_positions=[]) for site_id in site_ids}

    for site_structure in site_structures.values():
        site_id = site_structure['_struct_site_gen.site_id']

        if auth_id:
            chain_id = site_structure['_struct_site_gen.auth_asym_id']
            res_id = int(site_structure['_struct_site_gen.auth_seq_id'])
            res_name = site_structure['_struct_site_gen.auth_comp_id']      # Residue CCD ID.
        else:
            # [NOTE]:
            #   We convert mmcif chain id into author chain id to align with `af2_mmcif_parsing`.
            #   (So that we can match binding site positions with output FASTA)
            mmcif_chain_id = site_structure['_struct_site_gen.label_asym_id']
            try:
                chain_id = mmcif_object.mmcif_to_author_chain_id[mmcif_chain_id]
            except KeyError:
                print(f'chain id {mmcif_chain_id} not in the target region, skipping it.')
                continue
            res_id = site_structure['_struct_site_gen.label_seq_id']
            if is_empty(res_id):
                # [NOTE]: We skip empty residue positions directly (not in final positions).
                continue
            else:
                res_id = int(res_id)
            res_name = site_structure['_struct_site_gen.label_comp_id']
        insertion_code = norm_cif_empty_value(site_structure['_struct_site_gen.pdbx_auth_ins_code'])

        result[site_id].binding_positions.append(SingleBindingPosition(
            chain_id=chain_id,
            res_id=res_id,
            res_name=res_name,
            insertion_code=insertion_code,
        ))

    for site in result.values():
        site.binding_positions.sort()

    return result


def get_bio_chain(mmcif_object: AF2MmcifObject, auth_chain_id: str) -> PdbChain:
    """Get a Biopython chain object.

    Args:
        mmcif_object:
        auth_chain_id:

    Returns:

    """
    relevant_chains = [c for c in mmcif_object.structure.get_chains()
                       if c.id == auth_chain_id]
    if len(relevant_chains) != 1:
        raise MultipleChainsError(
            f'Expected exactly one chain in structure with id {auth_chain_id}.')
    return relevant_chains[0]


def get_bio_residue(
        mmcif_object: AF2MmcifObject, auth_chain_id: str, res_index: int,
        chain: PdbChain = None,
) -> Tuple[ResidueAtPosition, Optional[PdbResidue]]:
    """Get a single Biopython residue object.

    Args:
        mmcif_object:
        auth_chain_id:
        res_index:
        chain: Optional, provide the chain to index directly (used by outer iterators)

    Returns:

    """
    if chain is None:
        chain = get_bio_chain(mmcif_object, auth_chain_id)

    res_at_position = mmcif_object.seqres_to_structure[auth_chain_id][res_index]
    if res_at_position.is_missing:
        return res_at_position, None
    else:
        res = chain[(res_at_position.hetflag,
                     res_at_position.position.residue_number,
                     res_at_position.position.insertion_code)]
        return res_at_position, res


def bio_residue_iterator(
        mmcif_object: AF2MmcifObject, auth_chain_id: str) -> Iterator[Tuple[ResidueAtPosition, Optional[PdbResidue]]]:
    """Iterate over Biopython residues safely.

    Args:
        mmcif_object:
        auth_chain_id:

    Returns:

    Yields:
        Tuple of (residue_at_position, Biopython residue object)
    """
    chain = get_bio_chain(mmcif_object, auth_chain_id)
    num_res = len(mmcif_object.chain_to_seqres[auth_chain_id])
    for res_index in range(num_res):
        yield get_bio_residue(mmcif_object, auth_chain_id, res_index, chain=chain)
