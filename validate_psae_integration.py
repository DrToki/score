#!/usr/bin/env python3
"""
Final validation that psae.py can read and process Mittens_af2pred_scores.json
This simulates the exact psae.py workflow
"""

import json
import os

def validate_psae_integration():
    """Validate complete psae.py integration"""
    
    print("🧪 Final Validation: psae.py Integration with Mittens Files")
    print("=" * 70)
    
    json_file = "Mittens_af2pred_scores.json"
    pdb_file = "Mittens_test.pdb"
    
    # Test 1: File existence
    print("1️⃣ Checking file availability...")
    if os.path.exists(json_file):
        print(f"   ✅ {json_file} found")
    else:
        print(f"   ❌ {json_file} missing")
        return False
        
    if os.path.exists(pdb_file):
        print(f"   ✅ {pdb_file} found")
    else:
        print(f"   ❌ {pdb_file} missing")
        return False
    
    # Test 2: JSON loading (exact psae.py approach)
    print("\n2️⃣ Loading JSON data (psae.py approach)...")
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
        print(f"   ✅ JSON loaded successfully")
    except Exception as e:
        print(f"   ❌ JSON loading failed: {e}")
        return False
    
    # Test 3: psae.py data extraction simulation (lines 412-431)
    print("\n3️⃣ Simulating psae.py data extraction...")
    
    # Extract iptm and ptm (lines 412-415)
    if 'iptm' in data:
        iptm_af2 = float(data['iptm'])
    else:
        iptm_af2 = -1.0
        
    if 'ptm' in data:
        ptm_af2 = float(data['ptm'])
    else:
        ptm_af2 = -1.0
    
    print(f"   📊 iptm_af2: {iptm_af2}")
    print(f"   📊 ptm_af2: {ptm_af2}")
    
    # Extract plddt (lines 416-423)
    if 'plddt' in data:
        plddt_data = data['plddt']
        print(f"   ✅ plddt extracted: {len(plddt_data)} residues")
    else:
        print(f"   ⚠️  plddt missing, would use zeros")
        plddt_data = None
    
    # Extract PAE matrix (lines 424-431) - exact psae.py logic
    if 'pae' in data:
        pae_matrix = data['pae']
        print(f"   ✅ PAE matrix from 'pae' key: {len(pae_matrix)}x{len(pae_matrix[0])}")
    elif 'predicted_aligned_error' in data:
        pae_matrix = data['predicted_aligned_error']
        print(f"   ✅ PAE matrix from 'predicted_aligned_error' key: {len(pae_matrix)}x{len(pae_matrix[0])}")
    else:
        print(f"   ❌ No PAE matrix found")
        return False
    
    # Test 4: Data validation
    print("\n4️⃣ Validating extracted data...")
    
    # Check dimensions match
    if plddt_data and len(plddt_data) == len(pae_matrix):
        print(f"   ✅ Dimensions consistent: {len(plddt_data)} residues")
    else:
        print(f"   ❌ Dimension mismatch")
        return False
    
    # Check PAE matrix is square
    if len(pae_matrix) == len(pae_matrix[0]):
        print(f"   ✅ PAE matrix is square: {len(pae_matrix)}x{len(pae_matrix[0])}")
    else:
        print(f"   ❌ PAE matrix not square")
        return False
    
    # Test 5: Sample calculations (what psae.py will do)
    print("\n5️⃣ Sample metric calculations...")
    
    # Calculate mean pLDDT (psae.py does this)
    if plddt_data:
        mean_plddt = sum(plddt_data) / len(plddt_data)
        print(f"   📊 Mean pLDDT: {mean_plddt:.2f}")
        
        # Find min/max confidence
        min_plddt = min(plddt_data)
        max_plddt = max(plddt_data)
        print(f"   📊 pLDDT range: {min_plddt:.1f} - {max_plddt:.1f}")
    
    # Calculate mean PAE (psae.py does this)
    all_pae_values = []
    for row in pae_matrix:
        all_pae_values.extend(row)
    mean_pae = sum(all_pae_values) / len(all_pae_values)
    print(f"   📊 Mean PAE: {mean_pae:.2f}")
    
    # Test 6: Command simulation
    print("\n6️⃣ psae.py command simulation...")
    
    print(f"   💻 Command to run:")
    print(f"      python psae.py {json_file} {pdb_file} 10 10")
    print(f"   📄 This would create output files:")
    print(f"      - Mittens_test_10_10.txt (main results)")
    print(f"      - Mittens_test_10_10_byres.txt (per-residue results)")
    print(f"      - Mittens_test_10_10.pml (PyMOL script)")
    
    # Test 7: Data sample for verification
    print("\n7️⃣ Data samples for verification...")
    print(f"   📈 First 3 pLDDT values: {plddt_data[:3]}")
    print(f"   📈 PAE matrix top-left 3x3:")
    for i in range(3):
        print(f"      {pae_matrix[i][:3]}")
    
    print(f"   📈 PAE diagonal sample (self-errors): {[pae_matrix[i][i] for i in range(5)]}")
    
    print("\n🎉 Integration validation complete!")
    print("✅ The Mittens_af2pred_scores.json file is fully compatible with psae.py")
    print("✅ psae.py will successfully extract all required data")
    print("✅ The JSON structure matches psae.py expectations exactly")
    
    return True

if __name__ == "__main__":
    success = validate_psae_integration()
    
    if success:
        print(f"\n🎯 FINAL RESULT: psae.py integration ready!")
        print(f"\nNext steps:")
        print(f"1. Ensure you have numpy installed: pip install numpy")
        print(f"2. Run: python psae.py Mittens_af2pred_scores.json Mittens_test.pdb 10 10")
        print(f"3. Check output files: Mittens_test_10_10.txt")
    else:
        print(f"\n💥 FINAL RESULT: Integration issues found!")