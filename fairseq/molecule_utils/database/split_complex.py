#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Split the target-ligand complex in the given PDB file."""

import dataclasses
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from Bio.PDB import PDBIO, Select, MMCIFIO
from Bio.PDB.Structure import Structure as PdbStructure
from dataclasses import dataclass, asdict

from . import structure_file, af2_mmcif_parsing
from .common_utils import check_ext, is_empty, norm_cif_empty_value
from .mmcif_utils import get_uniprot_ref
from .pdb_helper_datasets import get_pdb_ccd_info, CcdInfo
from .. import config
from ..exceptions import AF2MmCIFParseError


@dataclass(order=True)
class LigandInfo:
    ligand_id: int
    pdb_id: str
    ccd_id: str
    model_id: int
    chain_id: str
    res_id: int
    insertion_code: str


def _get_ligand_info(pdb_id: str, structure: PdbStructure, all_ccd_info) -> List[LigandInfo]:
    """Get ligand information of this structure."""
    ligand_info = []

    for model in structure:
        model_id = model.id
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                hetero_flag = residue.id[0]
                if hetero_flag.startswith('H_'):
                    ligand_info.append(LigandInfo(
                        ligand_id=len(ligand_info),
                        pdb_id=pdb_id,
                        ccd_id=hetero_flag[2:],
                        model_id=model_id,
                        chain_id=chain_id,
                        res_id=residue.id[1],
                        insertion_code=residue.id[2],
                    ))

    # TODO: Filter some ligands?

    return ligand_info


class MainTargetSelect(Select):
    """Select main target component of the structure (remove water and hetero residues)."""
    def accept_residue(self, residue):
        res = residue.id[0]
        if res == ' ':
            # ' ' means main target residue.
            return 1
        else:
            # 'W' means water molecule.
            # 'H_XXX' means hetero residue.
            return 0


class LigandSelect(Select):
    def __init__(self, model_id, chain_id, seq_id):
        super().__init__()
        self.model_id = model_id
        self.chain_id = chain_id
        self.seq_id = seq_id

    def accept_model(self, model):
        return 1 if model.id == self.model_id else 0

    def accept_chain(self, chain):
        return 1 if chain.id == self.chain_id else 0

    def accept_residue(self, residue):
        hetatm_flag, resseq, icode = residue.get_id()
        if not hetatm_flag.startswith('H_'):
            return 0
        if icode != ' ':
            logging.warning('Icode %s at position %s', icode, resseq)
        if resseq == self.seq_id:
            return 1
        else:
            return 0


def _save_structure(filename, structure, select, ext):
    if ext == '.cif':
        io = MMCIFIO()
    else:
        io = PDBIO()
    io.set_structure(structure)
    try:
        io.save(str(filename), select=select)
    except TypeError as e:
        # Clean bad file.
        if filename.exists():
            filename.unlink()
        if str(e) == '%c requires int or char':
            raise RuntimeError(f'Chain id is not a single char') from e
        else:
            raise


def _dump_split_complex(
        pdb_id: str,
        structure: PdbStructure,
        ligand_info: List[LigandInfo],
        dest_path: Path,
        ext: str,
):

    # Save main target.
    main_select = MainTargetSelect()
    split_dest_filename = dest_path / f'{pdb_id}-no-ligand{ext}'
    _save_structure(split_dest_filename, structure, main_select, ext)

    # Save all ligands.
    for info in ligand_info:
        ligand_filename = dest_path / f'{pdb_id}-ligand{info.ligand_id}{ext}'
        ligand_select = LigandSelect(model_id=info.model_id, chain_id=info.chain_id, seq_id=info.res_id)
        _save_structure(ligand_filename, structure, ligand_select, ext)


@dataclass
class SplitComplexResult:
    """
    target_filename: Path to the target (without ligands and water).
    ligand_info_filename:
    ligand_filenames:
    ligand_info:
    """
    target_filename: Optional[Path]
    ligand_info_filename: Optional[Path]
    ligand_filenames: List[Path]
    ligand_info: List[LigandInfo]


def split_pdb_complex_paths(
        pdb_id: str,
        split_ext: str = '.pdb',
        split_cache_path: Path = None,
        pdb_cache_path: Path = None,
        ccd_cache_path: Path = None,
) -> SplitComplexResult:
    """Remove all heteros from the given PDB file, and return the split target and ligand file paths.

    Args:
        pdb_id:
        split_ext: File extension to store split structure files. Default: .pdb
            (NOTE: smina cannot parse .cif files with single value, so we fallback to .pdb format)
        split_cache_path (Path):
        pdb_cache_path:
        ccd_cache_path:

    Returns:
        SplitComplexResult: Dict of paths and information.

        Keys:
            target_filename (Path): Path to the target (without ligands and water).
            ligand_info_filename (Path):
            ligand_filenames (List[Path]):
            ligand_info (dict):
        All keys will be set to None or empty values if structure file not exists.

    Raises:
        FileNotFoundError is failed to get structure file.

    Notes:
        1. Internally use PdbStructure.
        2. Unlike `get_target_ligand_pairs_from_pdb`, this function will apply *NO* filtration.
        All hetero atoms will be remained (to get more possible docking inputs).
    """
    split_ext = check_ext(split_ext)

    pdb_id = pdb_id.lower()

    empty_result = SplitComplexResult(
        target_filename=None, ligand_info_filename=None,
        ligand_filenames=[], ligand_info=[],
    )

    if split_cache_path is None:
        split_cache_path = config.split_pdb_cache_path()
    if pdb_cache_path is None:
        pdb_cache_path = config.pdb_cache_path()
    if ccd_cache_path is None:
        ccd_cache_path = config.pdb_ccd_path()
    dest_path = split_cache_path / pdb_id[1:3] / pdb_id
    dest_path.mkdir(exist_ok=True, parents=True)

    def _ligand_filenames():
        return [
            dest_path / f'{pdb_id}-ligand{i}{split_ext}'
            for i in sorted(v.ligand_id for v in ligand_info)
        ]

    split_dest_filename = dest_path / f'{pdb_id}-no-ligand{split_ext}'
    ligand_info_filename = dest_path / f'{pdb_id}-ligand-info.json'
    if split_dest_filename.exists():
        assert ligand_info_filename.exists(), 'Ligand information filename not exists'
        with ligand_info_filename.open('r', encoding='utf-8') as f_info:
            ligand_info_dict = json.load(f_info)
            ligand_info = [LigandInfo(**d) for d in ligand_info_dict]
        return SplitComplexResult(
            target_filename=split_dest_filename,
            ligand_info_filename=ligand_info_filename,
            ligand_filenames=_ligand_filenames(),
            ligand_info=ligand_info,
        )

    try:
        structure = structure_file.get_pdb_structure(
            pdb_id, ext='.cif', pdb_cache_path=pdb_cache_path, mmcif_add_dict=True)
    except FileNotFoundError:
        logging.warning(f'Cannot find structure file of {pdb_id}.')
        return empty_result

    all_ccd_info = get_pdb_ccd_info(ccd_cache_path)
    ligand_info = _get_ligand_info(pdb_id, structure, all_ccd_info)
    with ligand_info_filename.open('w', encoding='utf-8') as f_info:
        ligand_info_dict = [asdict(info) for info in ligand_info]
        json.dump(ligand_info_dict, f_info, indent=4)

    _dump_split_complex(pdb_id, structure, ligand_info, dest_path=dest_path, ext=split_ext)

    return SplitComplexResult(
        target_filename=split_dest_filename,
        ligand_info_filename=ligand_info_filename,
        ligand_filenames=_ligand_filenames(),
        ligand_info=ligand_info,
    )


def split_pdb_complex_file(
        pdb_path: Path,
        split_save_path: Path = Path('.'),
        pdb_id: str = 'unknown',
        split_ext: str = '.pdb',
        ccd_cache_path: Path = None,
):
    """Split PDB file into split_save_path."""

    # TODO: Extract core code from `split_pdb_complex_paths` to here.


@dataclass
class SplitComplexPair:
    pdb_id: str
    chain_id: str
    res_id: int
    insertion_code: str
    site_ids: List[str]
    ligand_id: str
    chain_fasta: str
    ligand_inchi: str
    exist_in_uniprot: bool  # This ligand exists in one (or more) referenced UniProt chains.
    uniprot_ids: List[str]

    @classmethod
    def fields(cls):
        return [field.name for field in dataclasses.fields(cls)]

    @staticmethod
    def str2bool(s):
        if s == 'True':
            return True
        return False

    @staticmethod
    def str2list(s):
        if not s:
            return []
        return s.split(';')

    @classmethod
    def from_str_list(cls, str_list) -> 'SplitComplexPair':
        item = cls(*str_list)
        item.res_id = int(item.res_id)
        item.site_ids = cls.str2list(item.site_ids)
        item.exist_in_uniprot = cls.str2bool(item.exist_in_uniprot)
        item.uniprot_ids = cls.str2list(item.uniprot_ids)
        return item

    def to_str_list(self) -> List[str]:
        return [self.pdb_id, self.chain_id, str(self.res_id), self.insertion_code, ';'.join(self.site_ids),
                self.ligand_id, self.chain_fasta, self.ligand_inchi, str(self.exist_in_uniprot),
                ';'.join(self.uniprot_ids)]


_PAT_CARBON = re.compile(r'C |C$|C[0-9]+')


def _has_carbon(formula):
    match = _PAT_CARBON.search(formula)
    return match is not None


def _get_sites(
        mmcif_object: af2_mmcif_parsing.AF2MmcifObject,
        all_ccd_info: Dict[str, CcdInfo],
        query_chain_id: Optional[str],
):
    parsed_info = mmcif_object.raw_string
    pdb_id = mmcif_object.file_id

    # 1. Get all hetero items (some may not be ligands).
    #   [NOTE]: We only take non-polymers into consideration.
    ligands = af2_mmcif_parsing.mmcif_loop_to_dict('_chem_comp.', '_chem_comp.id', parsed_info)

    # 1.1 Filters
    #   1) Only accept non-polymer TODO: This may be relaxed in future (accept some polymers)?
    #   2) Remove some inorganic compounds (e.g. water)
    #   3) Must in CCD IDs
    #   4) Formula weight limited
    #   5) Remove all inorganic compounds (Carbon not in SMILES)
    filtered_ligands = {}
    for ccd_id, v in ligands.items():
        if v['_chem_comp.type'] != 'non-polymer':
            continue
        if ccd_id in config.SKIP_INORGANIC_LIGANDS:
            continue
        ccd_info = all_ccd_info.get(ccd_id, None)
        if ccd_info is None:
            continue
        if ccd_info.formula_weight > config.LIGAND_MAX_FORMULA_WEIGHT or math.isnan(ccd_info.formula_weight):
            continue
        if not _has_carbon(ccd_info.formula):
            continue
        filtered_ligands[ccd_id] = v
    ligands = filtered_ligands

    # 2. Get position of each ligand.
    # Key of `ligand_positions`: (ccd_id, chain_id, res_id, insertion_code)
    # [NOTE]:
    #   We use 'pdb_seq_num' (not 'auth_seq_num'), which is aligned to index mentioned in site details and
    #   other sections.
    # TODO: Need to change '_pdbx_nonpoly_scheme.pdb_seq_num' into '_pdbx_nonpoly_scheme.auth_seq_num'?
    # TODO: In some cases (e.g. 2dhb), pdb_mon_id != auth_mon_id, so we need to use pdb_mon_id (change from auth_mon_id) and rebuild the whole dataset.
    ligand_positions = af2_mmcif_parsing.mmcif_loop_to_dict(
        '_pdbx_nonpoly_scheme',
        ('_pdbx_nonpoly_scheme.auth_mon_id', '_pdbx_nonpoly_scheme.pdb_strand_id', '_pdbx_nonpoly_scheme.pdb_seq_num',
         '_pdbx_nonpoly_scheme.pdb_ins_code'),
        parsed_info)
    new_ligand_positions = {}
    for k, v in ligand_positions.items():
        ccd_id, chain_id, seq_id, insertion_code = k
        if query_chain_id is not None and ccd_id != query_chain_id:
            continue
        if ccd_id not in ligands:
            continue
        try:
            seq_num = int(seq_id)
        except ValueError:
            logging.warning(f'Cannot parse pdb_seq_num {k[2]}')
            continue
        insertion_code = norm_cif_empty_value(insertion_code)
        new_ligand_positions[(ccd_id, chain_id, seq_num, insertion_code)] = v
    ligand_positions = new_ligand_positions

    # 3. Get ligand - binding site relation (L2BS) information.
    ligand2site = {position: set() for position in ligand_positions}

    # 3.1. Get binding sites structures from '_struct_site_gen', and try to get ligand information from them.
    site_structures = af2_mmcif_parsing.mmcif_loop_to_dict('_struct_site_gen.', '_struct_site_gen.id', parsed_info)

    for site_part in site_structures.values():
        # Match each site to previous ligand positions.
        # [NOTE]:
        #   We need to use 'auth_XXX' instead of 'label_XXX' here (to align author IDs of ligands).
        #   In `extract_binding_sites.py`, we need to use 'label_XXX'.
        ccd_id = site_part['_struct_site_gen.auth_comp_id']
        chain_id = site_part['_struct_site_gen.auth_asym_id']
        res_id = int(site_part['_struct_site_gen.auth_seq_id'])
        bind_position = (
            ccd_id, chain_id, res_id,
            norm_cif_empty_value(site_part['_struct_site_gen.pdbx_auth_ins_code']),
        )
        if bind_position in ligand_positions:
            site_id = site_part['_struct_site_gen.site_id']
            ligand2site[bind_position].add(site_id)

    # 3.2. In some mmCIF files, _struct_site will include some L2BS information. E.g., 6uap.cif.
    site_infos = af2_mmcif_parsing.mmcif_loop_to_dict('_struct_site.', '_struct_site.id', parsed_info)
    for site_id, site_info in site_infos.items():
        ligand_ccd_id = site_info['_struct_site.pdbx_auth_comp_id']
        if is_empty(ligand_ccd_id):
            continue
        ligand_chain_id = site_info['_struct_site.pdbx_auth_asym_id']
        if is_empty(ligand_chain_id):
            continue
        ligand_res_id = site_info['_struct_site.pdbx_auth_seq_id']
        if is_empty(ligand_res_id):
            continue
        ligand_res_id = int(ligand_res_id)
        ligand_insertion_code = norm_cif_empty_value(site_info['_struct_site.pdbx_auth_ins_code'])
        bind_position = (ligand_ccd_id, ligand_chain_id, ligand_res_id, ligand_insertion_code)
        if bind_position in ligand_positions:
            ligand2site[bind_position].add(site_id)

    # 3.3. Get L2BS information from details string. E.g., 4fpc.cif.
    for site_id, site_info in site_infos.items():
        details = site_info['_struct_site.details']
        bind_position = None
        if details.startswith('BINDING SITE FOR RESIDUE '):
            if not 31 <= len(details) <= 35:
                continue
            words = details.split()
            if len(words) != 7:
                continue
            ccd_id, chain_id, res_id, insertion_code = words[4], words[5], words[6], '.'
            try:
                res_id = int(res_id.strip())
            except ValueError:
                continue
            bind_position = (ccd_id, chain_id, res_id, insertion_code)
        # TODO: Other patterns.
        if bind_position is not None and bind_position in ligand_positions and not ligand2site[bind_position]:
            # [NOTE]: Evidence from details have low priority. We accept it only when all other methods have failed.
            ligand2site[bind_position].add(site_id)

    # TODO: 3.4. If still cannot get binding sites, try to create it from nearest residues.

    pdb_db_ref = get_uniprot_ref(mmcif_object)

    pairs = []
    for position in ligand_positions:
        ccd_id, chain_id, res_id, insertion_code = position
        if chain_id not in mmcif_object.chain_to_seqres:
            # [NOTE]: Case in 6yws and 4xqn.
            logging.warning(
                f'{pdb_id} chain {chain_id} not found. Maybe this ligand is bounded to a non-protein sequence.')
            continue

        # 4. Get related UniProt IDs.
        uniprot_ids = []
        exist_in_uniprot = False
        for ref in pdb_db_ref.values():
            if ref.chain_id == chain_id and ref.seq_begin <= res_id <= ref.seq_end:
                uniprot_ids.append(ref.uniprot_id)
                exist_in_uniprot = True
        if not uniprot_ids:
            # [NOTE]: We cannot specify the detailed chain position, so we record all mentioned UniProt IDs.
            uniprot_ids = [ref.uniprot_id for ref in pdb_db_ref.values()]

        pairs.append(SplitComplexPair(
            pdb_id=pdb_id,
            chain_id=chain_id,
            res_id=res_id,
            insertion_code=insertion_code,
            site_ids=sorted(ligand2site[position]),
            ligand_id=ccd_id,
            chain_fasta=mmcif_object.chain_to_seqres[chain_id],
            ligand_inchi=all_ccd_info[ccd_id].inchi,
            exist_in_uniprot=exist_in_uniprot,
            uniprot_ids=uniprot_ids,
        ))
    return pairs


def get_target_ligand_pairs_from_pdb(
        pdb_id: str,
        query_chain_id: Optional[str] = None,
        pdb_cache_path: Path = None,
        ccd_cache_path: Path = None,
) -> List[SplitComplexPair]:
    """Get target-ligand pairs from PDB file.

    Args:
        pdb_id:
        query_chain_id: If specified, will only return ligands on this chain.
        pdb_cache_path:
        ccd_cache_path:

    Returns:
        List[SplitComplexPair]: List of pairs in this PDB.

    Notes:
        Internally use AF2MmcifObject.
    """
    pdb_id = pdb_id.lower()

    if pdb_cache_path is None:
        pdb_cache_path = config.pdb_cache_path()
    if ccd_cache_path is None:
        ccd_cache_path = config.pdb_ccd_path()

    try:
        mmcif_object = structure_file.get_af2_mmcif_object(pdb_id, pdb_cache_path)
    except FileNotFoundError as e:
        logging.warning(f'Cannot find structure file of {pdb_id}.')
        return []
    except AF2MmCIFParseError as e:
        logging.warning(f'Fail to parse mmCIF file of {pdb_id}.')
        return []

    if mmcif_object is None:
        # No valid chains.
        logging.warning(f'No protein chains found in mmCIF file {pdb_id}.')
        return []

    all_ccd_info = get_pdb_ccd_info(ccd_cache_path)
    pairs = _get_sites(mmcif_object, all_ccd_info, query_chain_id=query_chain_id)

    return pairs
