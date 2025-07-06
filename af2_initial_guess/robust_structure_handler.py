#!/usr/bin/env python3
"""
Robust structure handling for AF2 predictions
Handles edge cases like PDB numbering, chain order, and structural irregularities
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Structure import Structure as BioPDBStructure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
import warnings
import re
from simple_structure import SimpleStructure

class RobustStructureHandler:
    """Enhanced structure handler that deals with real-world PDB edge cases"""
    
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        
    def auto_detect_binder_target(self, structure: SimpleStructure, 
                                 binder_length_threshold: int = 150) -> Tuple[int, int]:
        """
        Auto-detect which chain is binder vs target based on length
        Shorter chain is typically the designed binder
        """
        chains = structure.split_by_chain()
        if len(chains) != 2:
            raise ValueError(f"Expected 2 chains, found {len(chains)}. Use manual chain specification.")
        
        chain_lengths = [(i, chain.size()) for i, chain in enumerate(chains)]
        chain_lengths.sort(key=lambda x: x[1])  # Sort by length
        
        # Shorter chain is likely the binder
        binder_idx, binder_len = chain_lengths[0]
        target_idx, target_len = chain_lengths[1]
        
        # Warn if binder seems too long
        if binder_len > binder_length_threshold:
            warnings.warn(f"Detected binder has {binder_len} residues (>{binder_length_threshold}). "
                         f"This seems large for a designed binder. Consider manual specification.")
        
        return binder_idx, target_idx
    
    def renumber_structure(self, structure: SimpleStructure, 
                          binder_chain_id: Optional[str] = None,
                          target_chain_id: Optional[str] = None) -> SimpleStructure:
        """
        Renumber structure to have unique consecutive residue indices
        
        Args:
            structure: Input structure
            binder_chain_id: Specific chain ID for binder (if known)
            target_chain_id: Specific chain ID for target (if known)
        """
        chains = structure.split_by_chain()
        
        if len(chains) == 1:
            # Monomer - just renumber consecutively
            return self._renumber_single_chain(structure)
        
        elif len(chains) == 2:
            # Complex - identify binder and target
            if binder_chain_id and target_chain_id:
                binder_idx, target_idx = self._find_chains_by_id(chains, binder_chain_id, target_chain_id)
            else:
                binder_idx, target_idx = self.auto_detect_binder_target(structure)
            
            return self._renumber_complex(chains, binder_idx, target_idx)
        
        else:
            raise ValueError(f"Cannot handle {len(chains)} chains. Maximum supported: 2")
    
    def _renumber_single_chain(self, structure: SimpleStructure) -> SimpleStructure:
        """Renumber single chain consecutively starting from 1"""
        new_structure = structure.structure.copy()
        
        res_num = 1
        for model in new_structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Standard residue
                        residue.id = (' ', res_num, ' ')
                        res_num += 1
        
        # Create temporary file for renumbered structure
        temp_file = f"/tmp/renumbered_{structure.pdb_file.stem}.pdb"
        io = PDBIO()
        io.set_structure(new_structure)
        io.save(temp_file)
        
        return SimpleStructure(temp_file)
    
    def _renumber_complex(self, chains: List[SimpleStructure], 
                         binder_idx: int, target_idx: int) -> SimpleStructure:
        """Renumber complex with binder first, then target"""
        binder_chain = chains[binder_idx]
        target_chain = chains[target_idx]
        
        # Renumber binder starting from 1
        binder_renumbered = self._renumber_chain_section(binder_chain, start_num=1)
        binder_end = binder_renumbered.size()
        
        # Renumber target starting after binder
        target_renumbered = self._renumber_chain_section(target_chain, start_num=binder_end + 1)
        
        # Combine chains
        combined_structure = self._combine_chains(binder_renumbered, target_renumbered)
        
        return combined_structure
    
    def _renumber_chain_section(self, chain_structure: SimpleStructure, start_num: int) -> SimpleStructure:
        """Renumber a single chain starting from specified number"""
        new_structure = chain_structure.structure.copy()
        
        res_num = start_num
        for model in new_structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Standard residue
                        residue.id = (' ', res_num, ' ')
                        res_num += 1
        
        # Create temporary file
        temp_file = f"/tmp/chain_renumbered_{start_num}.pdb"
        io = PDBIO()
        io.set_structure(new_structure)
        io.save(temp_file)
        
        return SimpleStructure(temp_file)
    
    def _combine_chains(self, binder: SimpleStructure, target: SimpleStructure) -> SimpleStructure:
        """Combine binder and target into single structure"""
        # Create new structure
        combined_structure = BioPDBStructure("combined")
        model = Model(0)
        combined_structure.add(model)
        
        # Add binder chain (rename to A)
        binder_chain = list(binder.structure.get_chains())[0].copy()
        binder_chain.id = 'A'
        model.add(binder_chain)
        
        # Add target chain (rename to B)  
        target_chain = list(target.structure.get_chains())[0].copy()
        target_chain.id = 'B'
        model.add(target_chain)
        
        # Save combined structure
        temp_file = "/tmp/combined_structure.pdb"
        io = PDBIO()
        io.set_structure(combined_structure)
        io.save(temp_file)
        
        return SimpleStructure(temp_file)
    
    def _find_chains_by_id(self, chains: List[SimpleStructure], 
                          binder_id: str, target_id: str) -> Tuple[int, int]:
        """Find chain indices by chain ID"""
        binder_idx = target_idx = None
        
        for i, chain_struct in enumerate(chains):
            chain_id = list(chain_struct.structure.get_chains())[0].get_id()
            if chain_id == binder_id:
                binder_idx = i
            elif chain_id == target_id:
                target_idx = i
        
        if binder_idx is None:
            raise ValueError(f"Binder chain '{binder_id}' not found")
        if target_idx is None:
            raise ValueError(f"Target chain '{target_id}' not found")
            
        return binder_idx, target_idx
    
    def validate_structure(self, structure: SimpleStructure) -> Dict[str, any]:
        """
        Comprehensive structure validation
        Returns validation report with warnings and errors
        """
        report = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'chain_info': [],
            'residue_gaps': [],
            'missing_atoms': []
        }
        
        chains = structure.split_by_chain()
        
        # Check chain count
        if len(chains) == 0:
            report['errors'].append("No chains found in structure")
            report['valid'] = False
            return report
        elif len(chains) > 2:
            report['warnings'].append(f"Found {len(chains)} chains. Only first 2 will be used.")
        
        # Analyze each chain
        for i, chain in enumerate(chains):
            chain_info = {
                'index': i,
                'length': chain.size(),
                'sequence': chain.sequence(),
                'residue_numbers': [res['number'] for res in chain.residues]
            }
            
            # Check for residue gaps
            res_numbers = chain_info['residue_numbers']
            if res_numbers:
                gaps = []
                for j in range(len(res_numbers) - 1):
                    if res_numbers[j+1] - res_numbers[j] > 1:
                        gaps.append((res_numbers[j], res_numbers[j+1]))
                
                if gaps:
                    chain_info['gaps'] = gaps
                    report['residue_gaps'].extend(gaps)
                    report['warnings'].append(f"Chain {i}: Found residue gaps: {gaps}")
            
            # Check for duplicate residue numbers
            if len(set(res_numbers)) != len(res_numbers):
                duplicates = [x for x in set(res_numbers) if res_numbers.count(x) > 1]
                report['errors'].append(f"Chain {i}: Duplicate residue numbers: {duplicates}")
                report['valid'] = False
            
            report['chain_info'].append(chain_info)
        
        # Check for overlapping residue numbers between chains
        if len(chains) == 2:
            chain1_nums = set(report['chain_info'][0]['residue_numbers'])
            chain2_nums = set(report['chain_info'][1]['residue_numbers'])
            overlap = chain1_nums & chain2_nums
            
            if overlap:
                report['errors'].append(f"Overlapping residue numbers between chains: {sorted(overlap)}")
                report['valid'] = False
        
        return report
    
    def clean_structure(self, structure: SimpleStructure, 
                       remove_waters: bool = True,
                       remove_hetero: bool = False,
                       keep_only_ca: bool = False) -> SimpleStructure:
        """
        Clean structure by removing unwanted components
        
        Args:
            remove_waters: Remove water molecules
            remove_hetero: Remove hetero atoms (ligands, etc.)
            keep_only_ca: Keep only CA atoms (for backbone-only predictions)
        """
        
        class CleanSelect(Select):
            def accept_residue(self, residue):
                # Remove water
                if remove_waters and residue.get_resname() in ['HOH', 'WAT']:
                    return False
                
                # Remove hetero atoms
                if remove_hetero and residue.get_id()[0] != ' ':
                    return False
                
                return True
            
            def accept_atom(self, atom):
                # Keep only CA atoms if requested
                if keep_only_ca:
                    return atom.get_name() == 'CA'
                
                # Standard protein atoms
                return atom.get_name() in ['N', 'CA', 'C', 'O', 'CB']
        
        # Apply cleaning
        temp_file = "/tmp/cleaned_structure.pdb"
        io = PDBIO()
        io.set_structure(structure.structure)
        io.save(temp_file, CleanSelect())
        
        return SimpleStructure(temp_file)

def prepare_structure_for_af2(pdb_file: Union[str, Path],
                             binder_chain: Optional[str] = None,
                             target_chain: Optional[str] = None,
                             auto_clean: bool = True,
                             auto_renumber: bool = True) -> SimpleStructure:
    """
    High-level function to prepare any PDB structure for AF2 prediction
    
    Args:
        pdb_file: Path to PDB file
        binder_chain: Chain ID of binder (if known)
        target_chain: Chain ID of target (if known)  
        auto_clean: Remove waters and non-standard residues
        auto_renumber: Automatically renumber residues for AF2 compatibility
        
    Returns:
        Cleaned and renumbered SimpleStructure ready for AF2
    """
    
    handler = RobustStructureHandler()
    
    # Load structure
    structure = SimpleStructure(str(pdb_file))
    print(f"📁 Loaded structure: {pdb_file}")
    print(f"   Chains: {len(structure.split_by_chain())}")
    print(f"   Total residues: {structure.size()}")
    
    # Validate structure
    validation = handler.validate_structure(structure)
    
    if not validation['valid']:
        print("❌ Structure validation failed:")
        for error in validation['errors']:
            print(f"   Error: {error}")
        raise ValueError("Structure has critical errors. Please fix manually.")
    
    if validation['warnings']:
        print("⚠️  Structure warnings:")
        for warning in validation['warnings']:
            print(f"   Warning: {warning}")
    
    # Clean structure if requested
    if auto_clean:
        print("🧹 Cleaning structure...")
        structure = handler.clean_structure(structure, 
                                          remove_waters=True,
                                          remove_hetero=True)
        print(f"   Residues after cleaning: {structure.size()}")
    
    # Renumber structure if requested
    if auto_renumber:
        print("🔢 Renumbering structure...")
        structure = handler.renumber_structure(structure, binder_chain, target_chain)
        
        # Validate renumbering
        validation_after = handler.validate_structure(structure)
        if validation_after['valid']:
            print("✅ Structure successfully prepared for AF2")
            
            # Print chain summary
            for chain_info in validation_after['chain_info']:
                print(f"   Chain {chain_info['index']}: {chain_info['length']} residues "
                      f"(residues {min(chain_info['residue_numbers'])}-{max(chain_info['residue_numbers'])})")
        else:
            print("❌ Renumbering failed:")
            for error in validation_after['errors']:
                print(f"   Error: {error}")
    
    return structure