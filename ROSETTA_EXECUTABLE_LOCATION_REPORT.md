# Rosetta Executable Location Report

## Summary

The Rosetta executable for scoring is defined in multiple locations within `simple_pipeline.py` with configurable options through command-line arguments.

## Primary Locations

### 1. Default Value in SimpleScorer Constructor
**File**: `simple_pipeline.py:97`
```python
def __init__(self, rosetta_path: str = "rosetta_scripts", xml_script: str = None, 
             use_ipsae: bool = True, af2_predictions_dir: str = "af2_predictions",
             rosetta_database_dir: str = None, auto_clean: bool = True, 
             auto_renumber: bool = True, strict_validation: bool = False):
    self.rosetta_path = rosetta_path  # Line 101
```

**Default Value**: `"rosetta_scripts"`

### 2. Command-Line Argument Definition
**File**: `simple_pipeline.py:799`
```python
parser.add_argument("--rosetta_path", default="rosetta_scripts", help="Rosetta executable")
```

**Default Value**: `"rosetta_scripts"`
**Help Text**: "Rosetta executable"

### 3. Argument Passing to SimpleScorer
**File**: `simple_pipeline.py:813`
```python
scorer = SimpleScorer(
    rosetta_path=args.rosetta_path,  # Uses command-line argument
    xml_script=args.xml_script, 
    use_ipsae=not args.no_ipsae, 
    af2_predictions_dir="af2_predictions",
    rosetta_database_dir=args.rosetta_database_dir,
    auto_clean=not args.no_auto_clean,
    auto_renumber=not args.no_auto_renumber,
    strict_validation=args.strict_validation
)
```

### 4. Usage in Rosetta Execution
**File**: `simple_pipeline.py:549`
```python
cmd = [
    self.rosetta_path,  # Uses the configured path
    '-s', pdb_file,
    '-parser:protocol', self.xml_script,
    '-out:file:scorefile', str(score_file),
    # ... other arguments
]
```

## Configuration Methods

### Method 1: Command-Line Override
```bash
python simple_pipeline.py --pdb complex.pdb --rosetta_path /path/to/rosetta_scripts
```

### Method 2: Full Path Specification
```bash
python simple_pipeline.py --pdb complex.pdb --rosetta_path /usr/local/rosetta/bin/rosetta_scripts.default.linuxgccrelease
```

### Method 3: Environment PATH
If `rosetta_scripts` is in your PATH, the default will work:
```bash
export PATH=$PATH:/path/to/rosetta/bin
python simple_pipeline.py --pdb complex.pdb
```

### Method 4: Programmatic Configuration
```python
from simple_pipeline import SimpleScorer

scorer = SimpleScorer(
    rosetta_path="/custom/path/to/rosetta_scripts",
    xml_script="my_protocol.xml"
)
```

## Related Configuration

### Rosetta Database Directory
**File**: `simple_pipeline.py:105`
```python
self.rosetta_database_dir = rosetta_database_dir or "/net/software/Rosetta/main/database"
```

**Default**: `"/net/software/Rosetta/main/database"`
**Command-line**: `--rosetta_database_dir`

### XML Script Configuration
**File**: `simple_pipeline.py:102`
```python
self.xml_script = xml_script  # Will be provided later
```

**Command-line**: `--xml_script`
**Example XML**: `configs/interface_scoring.xml`

## Execution Context

### When Rosetta is Called
**File**: `simple_pipeline.py:525-578`

The `_run_rosetta()` method:
1. Checks if `xml_script` is provided (line 528)
2. If no XML script: returns mock scores (lines 530-534)
3. If XML script provided: executes Rosetta command (lines 548-573)

### Mock Behavior (No XML Script)
```python
if not self.xml_script:
    # Return mock scores until XML is provided
    return RosettaScores(
        total_score=-500.0,
        interface_score=-20.0,
        binding_energy=-15.0
    )
```

### Actual Rosetta Execution
```python
cmd = [
    self.rosetta_path,           # rosetta_scripts or custom path
    '-s', pdb_file,             # Input PDB
    '-parser:protocol', self.xml_script,  # XML protocol
    '-out:file:scorefile', str(score_file),  # Output scores
    '-overwrite',
    # ... additional flags
]

subprocess.run(cmd, check=True, timeout=3600)
```

## Installation Documentation

### In INSTALL.md (Line 149)
```bash
# Manual specification
python simple_pipeline.py --pdb test.pdb --rosetta_path /full/path/to/rosetta_scripts
```

### PATH Check Example (Line 146)
```bash
# Check PATH
echo $PATH | grep rosetta
```

## Example Usage Patterns

### 1. Standard Installation
```bash
# Rosetta installed in standard location, added to PATH
python simple_pipeline.py --pdb complex.pdb --xml_script interface_scoring.xml
```

### 2. Custom Installation
```bash
# Rosetta installed in custom location
python simple_pipeline.py \
    --pdb complex.pdb \
    --xml_script interface_scoring.xml \
    --rosetta_path /opt/rosetta/bin/rosetta_scripts.default.linuxgccrelease \
    --rosetta_database_dir /opt/rosetta/database
```

### 3. HPC Environment
```bash
# Module-based HPC environment
module load rosetta
python simple_pipeline.py --pdb complex.pdb --xml_script interface_scoring.xml
```

### 4. Container Environment
```bash
# Using Rosetta container
singularity exec rosetta.sif python simple_pipeline.py \
    --pdb complex.pdb \
    --xml_script interface_scoring.xml \
    --rosetta_path rosetta_scripts
```

## Troubleshooting

### Common Issues

1. **Rosetta Not Found**
   - Check if `rosetta_scripts` is in PATH
   - Use `--rosetta_path` to specify full path
   - Verify executable permissions

2. **Wrong Rosetta Version**
   - Specify exact binary (e.g., `rosetta_scripts.default.linuxgccrelease`)
   - Use `--rosetta_path` with full binary name

3. **Database Path Issues**
   - Use `--rosetta_database_dir` to specify correct database location
   - Default assumes `/net/software/Rosetta/main/database`

### Verification Commands
```bash
# Test if Rosetta is accessible
which rosetta_scripts

# Test direct execution
rosetta_scripts -help

# Test with full path
/path/to/rosetta_scripts -help
```

## Configuration Priority

1. **Command-line argument** (`--rosetta_path`) - Highest priority
2. **Environment PATH** - If `rosetta_scripts` is found in PATH
3. **Default value** (`"rosetta_scripts"`) - Fallback

## Summary

**The Rosetta executable path is primarily configured through:**
- **Default**: `"rosetta_scripts"` (assumes it's in PATH)
- **Command-line**: `--rosetta_path /custom/path`
- **Programmatic**: `SimpleScorer(rosetta_path="/path")`

**The executable is used when:**
- An XML script is provided (`--xml_script`)
- Otherwise, mock scores are returned

**For production use, typically specify:**
```bash
--rosetta_path /full/path/to/rosetta_scripts.binary
--rosetta_database_dir /full/path/to/rosetta/database
```