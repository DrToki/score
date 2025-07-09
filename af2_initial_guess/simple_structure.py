#!/usr/bin/env python3
"""
Simple structure handling without PyRosetta dependency
Uses BioPython for PDB parsing and manipulation
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Structure import Structure as BioPDBStructure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class SimpleStructure:
    """Simple structure class to replace PyRosetta pose functionality"""
    
    def __init__(self, pdb_file: str):
        """Initialize from PDB file"""
        self.pdb_file = Path(pdb_file)
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure('protein', str(pdb_file))
        
        # Initialize original_chain_id to None (will be set by split_by_chain if needed)
        self.original_chain_id = None
        
        # Extract basic information
        self.sequence_str = self._extract_sequence()
        self.atoms = self._extract_atoms()
        self.residues = self._extract_residues()
        
    def _extract_sequence(self) -> str:
        """Extract amino acid sequence from structure"""
        aa_dict = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        sequence = ""
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Standard residue
                        resname = residue.get_resname()
                        if resname in aa_dict:
                            sequence += aa_dict[resname]
                        else:
                            sequence += 'X'  # Unknown residue
        
        return sequence
    
    def _extract_atoms(self) -> np.ndarray:
        """Extract atom coordinates as numpy array"""
        coords = []
        for atom in self.structure.get_atoms():
            coords.append(atom.get_coord())
        return np.array(coords)
    
    def _extract_residues(self) -> List[Dict]:
        """Extract residue information"""
        residues = []
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Standard residue
                        res_info = {
                            'name': residue.get_resname(),
                            'number': residue.get_id()[1],
                            'chain': chain.get_id(),
                            'atoms': [atom.get_name() for atom in residue.get_atoms()]
                        }
                        residues.append(res_info)
        return residues
    
    def sequence(self) -> str:
        """Get sequence string (compatible with PyRosetta pose.sequence())"""
        return self.sequence_str
    
    def size(self) -> int:
        """Get number of residues (compatible with PyRosetta pose.size())"""
        return len([r for r in self.structure.get_residues() if r.get_id()[0] == ' '])
    
    def total_residue(self) -> int:
        """Get total number of residues (compatible with PyRosetta pose.total_residue())"""
        return self.size()
    
    def split_by_chain(self) -> List['SimpleStructure']:
        """Split structure by chain (compatible with PyRosetta pose.split_by_chain())"""
        chains = []
        for model in self.structure:
            for chain in model:
                # Create new structure with single chain
                new_structure = BioPDBStructure(f"chain_{chain.get_id()}")
                new_model = Model(0)
                new_structure.add(new_model)
                
                # Copy chain to new structure, preserving original chain ID
                chain_copy = chain.copy()
                new_model.add(chain_copy)
                
                # Create temporary file for the chain
                temp_file = f"/tmp/chain_{chain.get_id()}_{self.pdb_file.stem}.pdb"
                io = PDBIO()
                io.set_structure(new_structure)
                io.save(temp_file)
                
                # Create SimpleStructure from temp file
                chain_struct = SimpleStructure(temp_file)
                # Store original chain ID for later reference
                chain_struct.original_chain_id = chain.get_id()
                chains.append(chain_struct)
        
        return chains
    
    def save(self, filename: str) -> None:
        """Save structure to PDB file"""
        io = PDBIO()
        io.set_structure(self.structure)
        io.save(filename)
    
    def dump_pdb(self, filename: str) -> None:
        """Save structure to PDB file (compatible with PyRosetta pose.dump_pdb())"""
        self.save(filename)
    
    def get_atoms_for_af2(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract atoms in format needed for AF2 (positions and masks)"""
        from alphafold.common import residue_constants
        
        # Initialize arrays
        num_residues = self.size()
        atom_positions = np.zeros((num_residues, residue_constants.atom_type_num, 3))
        atom_masks = np.zeros((num_residues, residue_constants.atom_type_num))
        
        # Fill arrays
        res_idx = 0
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Standard residue
                        for atom in residue:
                            atom_name = atom.get_name()
                            if atom_name in residue_constants.atom_types:
                                atom_idx = residue_constants.atom_order[atom_name]
                                atom_positions[res_idx, atom_idx] = atom.get_coord()
                                atom_masks[res_idx, atom_idx] = 1.0
                        res_idx += 1
        
        return atom_positions, atom_masks
    
    def get_chain_breaks(self, max_distance: float = 3.0) -> List[int]:
        """Find chain breaks based on distance between consecutive residues"""
        breaks = []
        prev_c = None
        res_idx = 0
        
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Standard residue
                        try:
                            curr_n = residue['N'].get_coord()
                            if prev_c is not None:
                                distance = np.linalg.norm(curr_n - prev_c)
                                if distance > max_distance:
                                    breaks.append(res_idx)
                            prev_c = residue['C'].get_coord()
                        except KeyError:
                            # Missing backbone atoms, assume chain break
                            if prev_c is not None:
                                breaks.append(res_idx)
                        res_idx += 1
        
        return breaks
    
    def get_chain_id(self) -> str:
        """Get the chain ID of the first chain in the structure"""
        for model in self.structure:
            for chain in model:
                return chain.get_id()
        return 'A'  # Default fallback

def load_from_pdb_dir(pdb_dir: str) -> List[SimpleStructure]:
    """Load structures from PDB directory"""
    structures = []
    pdb_path = Path(pdb_dir)
    
    for pdb_file in pdb_path.glob('*.pdb'):
        try:
            structure = SimpleStructure(str(pdb_file))
            structures.append(structure)
        except Exception as e:
            print(f"Warning: Could not load {pdb_file}: {e}")
            continue
    
    return structures

def load_from_pdb_list(pdb_files: List[str]) -> List[SimpleStructure]:
    """Load structures from list of PDB files"""
    structures = []
    
    for pdb_file in pdb_files:
        try:
            structure = SimpleStructure(pdb_file)
            structures.append(structure)
        except Exception as e:
            print(f"Warning: Could not load {pdb_file}: {e}")
            continue
    
    return structures