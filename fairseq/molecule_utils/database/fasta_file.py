#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Get FASTA sequence file."""

import logging
from pathlib import Path
from typing import Optional, Union

import requests
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from .. import config


def download_fasta_file_from_uniprot(
        uniprot_id: str, uniprot_fasta_cache_path: Path = None) -> Optional[Path]:
    """Download PDB (or mmCIF) file from the UniProt website.

    Args:
        uniprot_id:
        uniprot_fasta_cache_path:

    Returns:
        Downloaded FASTA file path (if exist) or None if failed.
    """
    uniprot_id = uniprot_id.lower()

    if uniprot_fasta_cache_path is None:
        uniprot_fasta_cache_path = config.uniprot_fasta_cache_path()

    dest_filename = uniprot_fasta_cache_path / f'{uniprot_id}.fasta'
    if dest_filename.exists():
        return dest_filename

    url = config.GET_UNIPROT_FASTA_URL + f'/{uniprot_id}.fasta'
    resp = requests.get(url)
    if resp.status_code != 200:
        logging.warning(f'[ERROR]: Failed to retrieve {uniprot_id}.fasta. Cannot find FASTA file.')
        return None
    else:
        with dest_filename.open('w', encoding='utf-8') as f_fasta:
            f_fasta.write(resp.text)
        return dest_filename


def get_fasta_from_uniprot(
        uniprot_id: str, uniprot_fasta_cache_path: Path = None, get_str: bool = True
) -> Union[str, SeqRecord]:
    """Get FASTA sequence of given UniProt ID from UniProt FASTA file.

    Args:
        uniprot_id:
        uniprot_fasta_cache_path:
        get_str: If set to False, will return a Bio.SeqRecord.SeqRecord instance (contains extra information) instead.

    Returns:

    """
    uniprot_id = uniprot_id.lower()

    dest_filename = download_fasta_file_from_uniprot(uniprot_id, uniprot_fasta_cache_path)
    if dest_filename is None:
        raise FileNotFoundError(f'FASTA file of {uniprot_id} not found.')

    with dest_filename.open('r', encoding='utf-8') as f_fasta:
        records = list(SeqIO.parse(f_fasta, 'fasta'))
        assert len(records) == 1
        if get_str:
            return str(records[0].seq)
        else:
            return records[0]
