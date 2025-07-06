#!/usr/bin/env python3
"""
AF2 scoring without PyRosetta dependencies
Simplified version of predict.py that uses BioPython instead
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'af2_initial_guess'))

import numpy as np
import jax
import jax.numpy as jnp
from timeit import default_timer as timer

from alphafold.common import residue_constants
from alphafold.common import protein
from alphafold.common import confidence
from alphafold.data import pipeline
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model

import af2_util
from simple_structure import SimpleStructure

class AF2ScorerSimple:
    """Simplified AF2 scorer without PyRosetta"""
    
    def __init__(self, model_name: str = "model_1_ptm", num_recycles: int = 3):
        self.model_name = model_name
        self.num_recycles = num_recycles
        self._setup_model()
    
    def _setup_model(self):
        """Setup AF2 model (from existing predict.py)"""
        model_config = config.model_config(self.model_name)
        model_config.data.eval.num_ensemble = 1
        model_config.data.common.num_recycle = self.num_recycles
        model_config.model.num_recycle = self.num_recycles
        model_config.model.embeddings_and_evoformer.initial_guess = True
        model_config.data.common.max_extra_msa = 5
        model_config.data.eval.max_msa_clusters = 5
        
        params_dir = os.path.join(os.path.dirname(__file__), 'af2_initial_guess', 'model_weights')
        model_params = data.get_model_haiku_params(model_name=self.model_name, data_dir=params_dir)
        
        self.model_runner = model.RunModel(model_config, model_params)
    
    def score_structure(self, structure: SimpleStructure) -> dict:
        """Score structure with AF2"""
        
        sequence = structure.sequence()
        chains = structure.split_by_chain()
        
        # Determine if monomer or complex
        is_monomer = len(chains) == 1
        binder_length = chains[0].size() if not is_monomer else len(sequence)
        
        # Get atom positions using existing function
        all_atom_positions, all_atom_masks = structure.get_atoms_for_af2()
        
        # Generate features (simplified from predict.py)
        feature_dict, initial_guess = self._generate_features(
            sequence, all_atom_positions, all_atom_masks, is_monomer, binder_length
        )
        
        # Run AF2 prediction
        prediction_result = self.model_runner.apply(
            self.model_runner.params,
            jax.random.PRNGKey(0),
            feature_dict,
            initial_guess
        )
        
        # Extract scores
        return self._extract_scores(prediction_result, is_monomer, binder_length)
    
    def _generate_features(self, sequence: str, all_atom_positions: np.ndarray, 
                          all_atom_masks: np.ndarray, is_monomer: bool, binder_length: int):
        """Generate AF2 features (from predict.py logic)"""
        
        initial_guess = af2_util.parse_initial_guess(all_atom_positions)
        
        # Determine residue mask
        if is_monomer:
            residue_mask = [False for _ in range(len(sequence))]
        else:
            residue_mask = [i >= binder_length for i in range(len(sequence))]
        
        # Generate template features
        template_dict = af2_util.generate_template_features(
            sequence, all_atom_positions, all_atom_masks, residue_mask
        )
        
        # Create feature dictionary
        feature_dict = {
            **pipeline.make_sequence_features(sequence=sequence, description="none", num_res=len(sequence)),
            **pipeline.make_msa_features(msas=[[sequence]], deletion_matrices=[[[0]*len(sequence)]]),
            **template_dict
        }
        
        # Handle chain breaks
        if not is_monomer:
            breaks = af2_util.check_residue_distances(all_atom_positions, all_atom_masks, 3.0)
            feature_dict['residue_index'] = af2_util.insert_truncations(feature_dict['residue_index'], breaks)
        
        feature_dict = self.model_runner.process_features(feature_dict, random_seed=0)
        
        return feature_dict, initial_guess
    
    def _extract_scores(self, prediction_result: dict, is_monomer: bool, binder_length: int) -> dict:
        """Extract confidence scores from prediction"""
        
        # Calculate confidence scores
        confidences = {}
        confidences['plddt'] = confidence.compute_plddt(prediction_result['predicted_lddt']['logits'])
        
        if 'predicted_aligned_error' in prediction_result:
            confidences.update(confidence.compute_predicted_aligned_error(
                prediction_result['predicted_aligned_error']['logits'],
                prediction_result['predicted_aligned_error']['breaks']
            ))
        
        # Calculate summary scores
        plddt_array = confidences['plddt']
        plddt_total = np.mean(plddt_array)
        
        if is_monomer:
            plddt_binder = plddt_total
            pae_interaction = float('nan')
        else:
            plddt_binder = np.mean(plddt_array[:binder_length])
            pae = confidences.get('predicted_aligned_error', np.full((len(plddt_array), len(plddt_array)), np.nan))
            pae_interaction1 = np.mean(pae[:binder_length, binder_length:])
            pae_interaction2 = np.mean(pae[binder_length:, :binder_length])
            pae_interaction = (pae_interaction1 + pae_interaction2) / 2
        
        return {
            'plddt_total': float(plddt_total),
            'plddt_binder': float(plddt_binder),
            'pae_interaction': float(pae_interaction),
            'binder_length': binder_length,
            'is_monomer': is_monomer
        }

# Simple integration function
def score_pdb_with_af2(pdb_file: str) -> dict:
    """Simple function to score a PDB file with AF2"""
    structure = SimpleStructure(pdb_file)
    scorer = AF2ScorerSimple()
    return scorer.score_structure(structure)