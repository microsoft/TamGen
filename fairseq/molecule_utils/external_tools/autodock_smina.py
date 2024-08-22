#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Helper tool to call Autodock-smina."""

import contextlib
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple


@contextlib.contextmanager
def timing(msg: str):
    logging.info('Started %s', msg)
    tic = time.time()
    yield
    toc = time.time()
    logging.info('Finished %s in %.3f seconds', msg, toc - tic)


class SminaError(Exception):
    pass


class AutoDockSmina:
    def __init__(
            self, *,
            binary_path: Optional[Path] = None,
            exhaustiveness: int = 32,
            seed: int = 1234,
    ):
        if binary_path is None:
            binary_path = self.find_binary()
        if binary_path is None:
            raise RuntimeError('Must provide AutoDock-smina binary path.')
        self.binary_path = binary_path
        self.exhaustiveness = exhaustiveness
        self.seed = seed

    @staticmethod
    def find_binary() -> Optional[Path]:
        """Find smina executable."""
        if os.name == 'nt':
            process = subprocess.Popen(['where.exe', 'smina'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            process = subprocess.Popen(['which', 'smina'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

    def _do_query(self, cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with timing(f'AutoDock-smina query'):
            stdout, stderr = process.communicate()
            ret_code = process.wait()

        if ret_code:
            raise SminaError(f'AutoDock-smina failed\nstderr:\n{stderr.decode("utf-8")}\n')

        output = stdout.decode('utf-8')
        lines = output.splitlines()
        for i in range(len(lines) - 3):
            if lines[i].startswith('mode') and lines[i + 2].startswith('----'):
                target_line_index = i + 3
                break
        else:
            raise SminaError(f'Cannot find AutoDock-smina result\nstdout:\n{stdout.decode("utf-8")}\n')

        target_line = lines[target_line_index].strip()
        try:
            affinity = float(target_line.split()[1])
        except ValueError:
            raise SminaError(f'Cannot parse affinity value from line {target_line}')
        return affinity

    def query(
            self, receptor_path: Path, ligand_path: Path, autobox_ligand_path: Optional[Path] = None,
            output_complex_path: Optional[Path] = None,
    ) -> float:
        if autobox_ligand_path is None:
            autobox_ligand_path = receptor_path     # No autobox, use the whole structure directly.
        if output_complex_path is None:
            output_complex_path = Path('/dev/null')

        cmd = [
            str(self.binary_path),
            '--receptor', str(receptor_path),
            '--ligand', str(ligand_path),
            '--autobox_ligand', str(autobox_ligand_path),
            '--exhaustiveness', str(self.exhaustiveness),
            '--seed', str(self.seed),
            '--out', str(output_complex_path),
        ]
        return self._do_query(cmd)

    def query_box(
            self, receptor_path: Path, ligand_path: Path, center: Tuple[float, float, float],
            box: Tuple[float, float, float] = (20., 20., 20.), output_complex_path: Optional[Path] = None,
    ) -> float:
        """Run query with box coordinate."""
        if output_complex_path is None:
            output_complex_path = Path('/dev/null')
        cmd = [
            str(self.binary_path),
            '--receptor', str(receptor_path),
            '--ligand', str(ligand_path),
            '--center_x', str(center[0]), '--center_y', str(center[1]), '--center_z', str(center[2]),
            '--size_x', str(box[0]), '--size_y', str(box[1]), '--size_z', str(box[2]),
            '--exhaustiveness', str(self.exhaustiveness),
            '--seed', str(self.seed),
            '--out', str(output_complex_path),
        ]
        return self._do_query(cmd)
