#!/usr/bin/env python3
"""
Test script to verify chain parsing and renumbering fixes
"""

import tempfile
import os
from pathlib import Path
import sys

# Add the af2_initial_guess directory to the path
sys.path.insert(0, str(Path(__file__).parent / "af2_initial_guess"))

from simple_structure import SimpleStructure
from robust_structure_handler import RobustStructureHandler, prepare_structure_for_af2

def create_test_pdb():
    """Create a test PDB with two chains having overlapping residue numbers"""
    test_pdb_content = '''ATOM      1  N   ALA A   1      -8.901   4.127  -0.555  1.00 11.99           N  
ATOM      2  CA  ALA A   1      -8.608   3.135  -1.618  1.00 11.99           C  
ATOM      3  C   ALA A   1      -7.221   2.458  -1.897  1.00 11.99           C  
ATOM      4  O   ALA A   1      -6.632   2.596  -2.925  1.00 11.99           O  
ATOM      5  CB  ALA A   1      -9.018   3.789  -2.932  1.00 11.99           C  
ATOM      6  N   GLY A   2      -6.821   1.674  -0.941  1.00 11.99           N  
ATOM      7  CA  GLY A   2      -5.518   0.983  -0.959  1.00 11.99           C  
ATOM      8  C   GLY A   2      -4.334   1.853  -1.336  1.00 11.99           C  
ATOM      9  O   GLY A   2      -3.190   1.461  -1.509  1.00 11.99           O  
ATOM     10  N   VAL B   1      -4.563   3.132  -1.454  1.00 11.99           N  
ATOM     11  CA  VAL B   1      -3.512   4.073  -1.825  1.00 11.99           C  
ATOM     12  C   VAL B   1      -2.675   4.462  -0.605  1.00 11.99           C  
ATOM     13  O   VAL B   1      -3.123   4.751   0.478  1.00 11.99           O  
ATOM     14  CB  VAL B   1      -4.133   5.312  -2.483  1.00 11.99           C  
ATOM     15  CG1 VAL B   1      -3.099   6.271  -3.089  1.00 11.99           C  
ATOM     16  CG2 VAL B   1      -5.085   6.062  -1.534  1.00 11.99           C  
ATOM     17  N   LEU B   2      -1.390   4.448  -0.742  1.00 11.99           N  
ATOM     18  CA  LEU B   2      -0.483   4.794   0.350  1.00 11.99           C  
ATOM     19  C   LEU B   2       0.854   4.062   0.220  1.00 11.99           C  
ATOM     20  O   LEU B   2       1.154   3.235  -0.650  1.00 11.99           O  
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(test_pdb_content)
        return f.name

def test_chain_parsing():
    """Test that chain parsing preserves chain identities"""
    print("Testing chain parsing...")
    
    test_pdb = create_test_pdb()
    
    try:
        # Load structure
        structure = SimpleStructure(test_pdb)
        print(f"Original structure has {structure.size()} residues")
        
        # Split by chain
        chains = structure.split_by_chain()
        print(f"Split into {len(chains)} chains")
        
        # Check that original chain IDs are preserved
        for i, chain in enumerate(chains):
            if hasattr(chain, 'original_chain_id'):
                print(f"Chain {i}: original_chain_id = {chain.original_chain_id}")
            else:
                print(f"Chain {i}: No original_chain_id attribute")
        
        # Test renumbering
        print("\nTesting renumbering...")
        handler = RobustStructureHandler()
        
        # Test auto-detect
        try:
            binder_idx, target_idx = handler.auto_detect_binder_target(structure)
            print(f"Auto-detected binder: chain {binder_idx}, target: chain {target_idx}")
        except Exception as e:
            print(f"Auto-detection failed: {e}")
        
        # Test renumbering with chain IDs
        try:
            renumbered = handler.renumber_structure(structure, binder_chain='A', target_chain='B')
            print(f"Renumbered structure has {renumbered.size()} residues")
            
            # Check validation
            validation = handler.validate_structure(renumbered)
            print(f"Validation passed: {validation['valid']}")
            
            if validation['errors']:
                print("Errors:")
                for error in validation['errors']:
                    print(f"  - {error}")
                    
            if validation['warnings']:
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")
            
            # Test the key fix: check residue numbers after renumbering
            print("\nChecking residue numbering after renumbering:")
            for i, res in enumerate(renumbered.residues):
                print(f"  Residue {i+1}: {res['name']} {res['number']} chain {res['chain']}")
            
        except Exception as e:
            print(f"Renumbering failed: {e}")
            import traceback
            traceback.print_exc()
    
    finally:
        os.unlink(test_pdb)

def test_prepare_structure_for_af2():
    """Test the high-level prepare_structure_for_af2 function"""
    print("\n" + "="*50)
    print("Testing prepare_structure_for_af2...")
    
    test_pdb = create_test_pdb()
    
    try:
        prepared = prepare_structure_for_af2(
            pdb_file=test_pdb,
            binder_chain='A',
            target_chain='B',
            auto_clean=True,
            auto_renumber=True
        )
        
        print(f"Prepared structure has {prepared.size()} residues")
        
        # Check that we can split it again
        chains = prepared.split_by_chain()
        print(f"Can split into {len(chains)} chains")
        
        # Check residue numbers
        for i, res in enumerate(prepared.residues):
            print(f"Residue {i+1}: {res['name']} {res['number']} chain {res['chain']}")
        
    except Exception as e:
        print(f"prepare_structure_for_af2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        os.unlink(test_pdb)

def test_overlapping_residue_fix():
    """Test that overlapping residue numbers are properly handled"""
    print("\n" + "="*50)
    print("Testing overlapping residue number fix...")
    
    test_pdb = create_test_pdb()
    
    try:
        # Load structure
        structure = SimpleStructure(test_pdb)
        print(f"Original structure: {structure.size()} residues")
        
        # Show original residue numbers (both chains have 1,2,3 - overlapping!)
        print("Original residue numbers:")
        for i, res in enumerate(structure.residues):
            print(f"  Residue {i+1}: {res['name']} {res['number']} chain {res['chain']}")
        
        # Test validation before fix
        handler = RobustStructureHandler()
        validation = handler.validate_structure(structure)
        print(f"\nOriginal validation result: {validation['valid']}")
        if validation['errors']:
            print("Validation errors:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        # Now test that prepare_structure_for_af2 handles this gracefully
        print("\nTesting prepare_structure_for_af2 with overlapping residues...")
        try:
            prepared = prepare_structure_for_af2(
                pdb_file=test_pdb,
                binder_chain='A',
                target_chain='B',
                auto_clean=True,
                auto_renumber=True  # This should fix the overlapping residues
            )
            
            print(f"✅ Successfully prepared structure with {prepared.size()} residues")
            
            # Check that overlapping residues are fixed
            print("Final residue numbers after renumbering:")
            for i, res in enumerate(prepared.residues):
                print(f"  Residue {i+1}: {res['name']} {res['number']} chain {res['chain']}")
            
            # Verify no overlaps
            chain_a_nums = [res['number'] for res in prepared.residues if res['chain'] == 'A']
            chain_b_nums = [res['number'] for res in prepared.residues if res['chain'] == 'B']
            overlap = set(chain_a_nums) & set(chain_b_nums)
            
            if overlap:
                print(f"❌ Still have overlapping residue numbers: {overlap}")
            else:
                print("✅ No overlapping residue numbers - fix successful!")
            
        except Exception as e:
            print(f"❌ prepare_structure_for_af2 failed: {e}")
            import traceback
            traceback.print_exc()
    
    finally:
        os.unlink(test_pdb)

if __name__ == "__main__":
    test_chain_parsing()
    test_prepare_structure_for_af2()
    test_overlapping_residue_fix()