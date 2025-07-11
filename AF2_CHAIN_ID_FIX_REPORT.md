# AF2 Chain ID Fix Report

## Critical Issue Identified and Fixed ✅

**Problem**: AF2 prediction output had both binder and target on the same chain ID, preventing Rosetta from properly scoring protein-protein interfaces.

## Root Cause Analysis

### The Problem
The original AF2 prediction code in `simple_pipeline.py:309-311` used:
```python
# Save predicted structure
predicted_pdb_lines = protein.to_pdb(predicted_protein)
with open(af2_predicted_pdb, 'w') as f:
    f.write(predicted_pdb_lines)
```

**Issue**: `protein.to_pdb()` from AlphaFold outputs all residues with the same chain ID (typically 'A'), regardless of the fact that the input represents a binder-target complex.

### Impact on Rosetta Scoring
- **Rosetta XML scripts** expect distinct chain IDs to identify binder vs target
- **Interface analysis** requires chain A (binder) and chain B (target) 
- **Binding energy calculations** fail when both proteins are on the same chain
- **Residue selectors** in XML (e.g., `<Chain name="chainA" chains="A"/>`) don't work

### Example of the Problem
**Before Fix (AF2 output):**
```
ATOM      1  N   ALA A   1  ...  # Binder residue 1
ATOM     17  N   PHE A   4  ...  # Target residue 1 (wrong chain!)
ATOM     28  N   TRP A   5  ...  # Target residue 2 (wrong chain!)
```

**After Fix:**
```
ATOM      1  N   ALA A   1  ...  # Binder residue 1  
TER      17      A   3          # Chain terminator
ATOM     17  N   PHE B   4  ...  # Target residue 1 (correct chain!)
ATOM     28  N   TRP B   5  ...  # Target residue 2 (correct chain!)
```

## Solution Implemented

### 1. New Method: `_save_predicted_structure_with_chains()`
**Location**: `simple_pipeline.py:383-419`

**Purpose**: Replace the basic `protein.to_pdb()` with intelligent chain assignment

**Key Features**:
- Assigns chain A to binder residues (1 to binder_length)
- Assigns chain B to target residues (binder_length+1 to end)
- Handles monomers correctly (single chain A)
- Includes fallback to original method if fix fails

### 2. New Method: `_fix_chain_ids_in_pdb()`
**Location**: `simple_pipeline.py:421-495`

**Purpose**: Parse and modify PDB content to assign correct chain IDs

**Algorithm**:
1. Parse each ATOM/HETATM line
2. Extract residue number (1-indexed)
3. Assign chain A if residue ≤ binder_length, chain B otherwise
4. Add proper TER records between chains
5. Verify atom counts and chain assignment

### 3. Enhanced Integration
**Location**: `simple_pipeline.py:308-311`

**Change**: 
```python
# OLD:
predicted_pdb_lines = protein.to_pdb(predicted_protein)
with open(af2_predicted_pdb, 'w') as f:
    f.write(predicted_pdb_lines)

# NEW:
self._save_predicted_structure_with_chains(
    predicted_protein, af2_predicted_pdb, binder_length, is_monomer, tag
)
```

## Technical Implementation Details

### Chain Assignment Logic
```python
# For each ATOM line:
res_num = int(line[22:26].strip())  # Extract residue number

if res_num <= binder_length:
    chain_id = 'A'  # Binder
else:
    chain_id = 'B'  # Target

new_line = line[:21] + chain_id + line[22:]
```

### TER Record Insertion
```python
# Add TER after last binder residue
ter_line = f"TER   {binder_length + 1:4d}      A   {binder_length:4d}"
```

### Validation and Logging
```python
# Count atoms per chain for verification
binder_atoms = len([line for line in final_lines 
                   if line.startswith(('ATOM', 'HETATM')) and line[21] == 'A'])
target_atoms = len([line for line in final_lines 
                   if line.startswith(('ATOM', 'HETATM')) and line[21] == 'B'])

print(f"Chain A (binder): {binder_atoms} atoms")
print(f"Chain B (target): {target_atoms} atoms")
```

## Testing Results

### Test Case: 3-residue binder + 2-residue target
**Input**: 5 residues, all on chain A (binder_length = 3)

**Output**:
- Chain A: 16 atoms (residues 1-3) ✅
- Chain B: 25 atoms (residues 4-5) ✅  
- TER record: 1 (properly placed) ✅
- Total atoms: Preserved (41 → 41) ✅

### Validation Checks
- ✅ Binder residues correctly assigned to chain A
- ✅ Target residues correctly assigned to chain B
- ✅ Total atom count preserved
- ✅ Proper TER records separate chains
- ✅ PDB format compliance maintained

## Benefits for Rosetta Scoring

### 1. Interface Analysis Works
```xml
<!-- Rosetta XML can now properly select chains -->
<Chain name="chainA" chains="A"/>  <!-- Binder -->
<Chain name="chainB" chains="B"/>  <!-- Target -->
<Interface name="interface" chain1_selector="chainA" chain2_selector="chainB" distance="8.0"/>
```

### 2. Binding Energy Calculations
```xml
<!-- Can calculate binding energy between distinct chains -->
<InteractionEnergyMetric name="binding_energy" 
                        residue_selector1="chainA" 
                        residue_selector2="chainB" 
                        scorefxn="ref15"/>
```

### 3. Shape Complementarity
```xml
<!-- Interface shape complementarity analysis -->
<ShapeComplementarityMetric name="shape_complementarity" 
                           residue_selector1="interface_A" 
                           residue_selector2="interface_B"/>
```

### 4. Per-Chain Analysis
```xml
<!-- Separate scoring of each chain -->
<TotalEnergyMetric name="chainA_energy" residue_selector="chainA" scorefxn="ref15"/>
<TotalEnergyMetric name="chainB_energy" residue_selector="chainB" scorefxn="ref15"/>
```

## Error Handling and Fallbacks

### 1. Graceful Degradation
If chain ID fixing fails, the system falls back to original AF2 output:
```python
except Exception as e:
    print(f"Failed to save structure with chain IDs: {e}")
    print("Falling back to original AF2 output")
    predicted_pdb_lines = protein.to_pdb(predicted_protein)
    with open(output_file, 'w') as f:
        f.write(predicted_pdb_lines)
```

### 2. Monomer Handling
Monomers are correctly handled without modification:
```python
if is_monomer:
    # For monomers, save as single chain A
    predicted_pdb_lines = protein.to_pdb(predicted_protein)
    # ...
```

### 3. Comprehensive Logging
Detailed logging shows what's happening:
```
🔗 Assigning chain IDs: A (binder, residues 1-3), B (target, residues 4-5)
🔍 Chain assignment verification for test:
   Chain A (binder): 16 atoms
   Chain B (target): 25 atoms
📾 Saved structure with correct chain IDs (A=binder, B=target)
```

## Integration with Existing Pipeline

### 1. Chain Analysis Integration
The fix works seamlessly with the existing chain analysis:
```python
# Chain info already determined
chain_info = self._analyze_chain_structure(prepared_structure, tag)
binder_length = chain_info['binder_length']

# Chain ID fix uses this information
self._save_predicted_structure_with_chains(
    predicted_protein, af2_predicted_pdb, binder_length, is_monomer, tag
)
```

### 2. Validation Integration
The existing `_validate_chain_indices()` method ensures consistency throughout.

### 3. Template Mask Consistency
The chain ID assignment aligns with the template mask logic:
- Residues 1 to binder_length: Chain A (predicted freely)
- Residues binder_length+1 to end: Chain B (templated)

## Files Modified

1. **`simple_pipeline.py`**:
   - Line 308-311: Changed AF2 structure saving
   - Line 383-419: Added `_save_predicted_structure_with_chains()`
   - Line 421-495: Added `_fix_chain_ids_in_pdb()`

2. **`test_chain_id_fix.py`**: Comprehensive test demonstrating the fix

3. **`AF2_CHAIN_ID_FIX_REPORT.md`**: This documentation

## Verification Commands

### Manual Verification
```bash
# Run pipeline and check output PDB
python simple_pipeline.py --pdb complex.pdb --xml_script interface_scoring.xml

# Check chain IDs in AF2 prediction output
grep "^ATOM" af2_predictions/complex_af2pred.pdb | awk '{print $5}' | sort | uniq -c
```

### Expected Output
```
   # atoms A  # Chain A (binder)
   # atoms B  # Chain B (target)
```

## Result

**The AF2 prediction output now correctly assigns:**
- **Chain A**: Binder residues (designed protein)
- **Chain B**: Target residues (natural protein)
- **TER records**: Proper chain separation

**Impact**: Rosetta can now properly score protein-protein interfaces, calculate binding energies, and perform interface analysis.

**Backward Compatibility**: The fix maintains full compatibility with existing code and provides fallback mechanisms for edge cases.

This critical fix ensures that the AF2 + Rosetta pipeline produces scientifically meaningful interface scores rather than failing silently due to incorrect chain assignments.