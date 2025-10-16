"""
Food Safety Knowledge Graph (FSKG) Construction
Implements the heterogeneous graph with food categories and visual attributes
"""

import numpy as np
import networkx as nx
import torch
import json
from typing import Dict, List, Tuple, Optional


class FoodSafetyKnowledgeGraph:
    """Constructs and manages the Food Safety Knowledge Graph"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_foods = config['dataset']['num_food_categories']
        self.num_attributes = config['dataset']['num_visual_attributes']
        
        # Node sets
        self.food_categories = []
        self.visual_attributes = []
        
        # Adjacency matrices for different relation types
        self.adj_food_attr = None  # Food-Attribute relations
        self.adj_food_food = None  # Food-Food relations  
        self.adj_attr_attr = None  # Attribute-Attribute relations
        
    def load_from_files(self, 
                       food_file: str, 
                       attr_file: str, 
                       rel_file: str):
        """Load knowledge graph from JSON files"""
        
        # Load food categories
        with open(food_file, 'r') as f:
            food_data = json.load(f)
            self.food_categories = food_data['categories']
            
        # Load visual attributes
        with open(attr_file, 'r') as f:
            attr_data = json.load(f)
            self.visual_attributes = attr_data['attributes']
            
        # Load relationships
        with open(rel_file, 'r') as f:
            rel_data = json.load(f)
            
        # Build adjacency matrices
        self._build_adjacency_matrices(rel_data)
        
    def _build_adjacency_matrices(self, rel_data: Dict):
        """Construct adjacency matrices from relationship data"""
        
        num_nodes = self.num_foods + self.num_attributes
        
        # Initialize adjacency matrices
        self.adj_food_attr = np.zeros((self.num_foods, self.num_attributes))
        self.adj_food_food = np.zeros((self.num_foods, self.num_foods))
        self.adj_attr_attr = np.zeros((self.num_attributes, self.num_attributes))
        
        # Food-Attribute relations (Equation 1 from paper)
        for rel in rel_data['food_attribute']:
            food_idx = rel['food_id']
            attr_idx = rel['attribute_id']
            frequency = rel['frequency']
            importance = rel['importance']
            
            # w^FA_ij = α · p_ij · s_ij
            self.adj_food_attr[food_idx, attr_idx] = frequency * importance
            
        # Normalize
        alpha = 1.0 / (self.adj_food_attr.max() + 1e-8)
        self.adj_food_attr *= alpha
        
        # Food-Food relations (Equation 2 from paper)
        for rel in rel_data['food_food']:
            food_i = rel['food_id_1']
            food_j = rel['food_id_2']
            comp_sim = rel['compositional_similarity']
            hazard_sim = rel['hazard_similarity']
            proc_sim = rel['processing_similarity']
            
            # w^FF_ij = 0.4 · sim_comp + 0.4 · sim_hazard + 0.2 · sim_proc
            weight = 0.4 * comp_sim + 0.4 * hazard_sim + 0.2 * proc_sim
            self.adj_food_food[food_i, food_j] = weight
            self.adj_food_food[food_j, food_i] = weight
            
        # Attribute-Attribute co-occurrence relations
        for rel in rel_data['attribute_attribute']:
            attr_i = rel['attribute_id_1']
            attr_j = rel['attribute_id_2']
            cooccurrence = rel['cooccurrence']
            
            self.adj_attr_attr[attr_i, attr_j] = cooccurrence
            self.adj_attr_attr[attr_j, attr_i] = cooccurrence
            
    def get_full_adjacency_matrix(self) -> np.ndarray:
        """Construct full heterogeneous graph adjacency matrix"""
        
        num_nodes = self.num_foods + self.num_attributes
        adj_full = np.zeros((num_nodes, num_nodes))
        
        # Top-left: food-food connections
        adj_full[:self.num_foods, :self.num_foods] = self.adj_food_food
        
        # Top-right and bottom-left: food-attribute connections
        adj_full[:self.num_foods, self.num_foods:] = self.adj_food_attr
        adj_full[self.num_foods:, :self.num_foods] = self.adj_food_attr.T
        
        # Bottom-right: attribute-attribute connections
        adj_full[self.num_foods:, self.num_foods:] = self.adj_attr_attr
        
        return adj_full
    
    def get_normalized_adjacency(self, adj: np.ndarray) -> torch.Tensor:
        """Normalize adjacency matrix: A_tilde = D^(-1/2) * A * D^(-1/2)"""
        
        # Add self-connections
        adj = adj + np.eye(adj.shape[0])
        
        # Compute degree matrix
        degrees = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        # Normalize: D^(-1/2) * A * D^(-1/2)
        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        return torch.FloatTensor(adj_normalized)
    
    def get_class_attribute_matrix(self, 
                                   class_to_food_mapping: Dict[int, int]) -> np.ndarray:
        """
        Get attribute matrix for hazard classes based on food categories
        
        Args:
            class_to_food_mapping: Maps hazard class ID to primary food category ID
            
        Returns:
            Matrix of shape [num_classes, num_attributes]
        """
        
        num_classes = len(class_to_food_mapping)
        class_attr_matrix = np.zeros((num_classes, self.num_attributes))
        
        for class_id, food_id in class_to_food_mapping.items():
            class_attr_matrix[class_id] = self.adj_food_attr[food_id]
            
        return class_attr_matrix


class MultiSourceGraphs:
    """Manages multiple knowledge graph sources for MSGF module"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        
        # Three adjacency matrices for different knowledge sources
        self.adj_fskg = None      # Food Safety Knowledge Graph
        self.adj_hyperclass = None # Hyperclass hierarchy
        self.adj_cooccur = None    # Co-occurrence statistics
        
    def build_fskg_adjacency(self, 
                            fskg: FoodSafetyKnowledgeGraph,
                            class_to_food: Dict[int, int]) -> np.ndarray:
        """
        Build class-level adjacency from FSKG (Equation 6)
        A^1_ij = |A_i ∩ A_j| / |A_i ∪ A_j|
        """
        
        # Get attribute sets for each class
        class_attrs = fskg.get_class_attribute_matrix(class_to_food)
        
        # Compute Jaccard similarity
        adj = np.zeros((self.num_classes, self.num_classes))
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                attrs_i = set(np.where(class_attrs[i] > 0)[0])
                attrs_j = set(np.where(class_attrs[j] > 0)[0])
                
                if len(attrs_i | attrs_j) > 0:
                    adj[i, j] = len(attrs_i & attrs_j) / len(attrs_i | attrs_j)
                    
        self.adj_fskg = adj
        return adj
    
    def build_hyperclass_adjacency(self, 
                                  class_hierarchy: Dict[int, List[int]]) -> np.ndarray:
        """
        Build adjacency from class hierarchy (Equation 7)
        A^2_ij = level if classes share ancestor, 0 otherwise
        """
        
        adj = np.zeros((self.num_classes, self.num_classes))
        
        for class_i in range(self.num_classes):
            for class_j in range(self.num_classes):
                if class_i == class_j:
                    adj[class_i, class_j] = 1.0
                else:
                    # Find shared ancestor level
                    ancestors_i = set(class_hierarchy.get(class_i, []))
                    ancestors_j = set(class_hierarchy.get(class_j, []))
                    shared = ancestors_i & ancestors_j
                    
                    if shared:
                        # Use highest shared level
                        adj[class_i, class_j] = len(shared) / max(len(ancestors_i), len(ancestors_j))
                        
        self.adj_hyperclass = adj
        return adj
    
    def build_cooccurrence_adjacency(self, 
                                    cooccurrence_counts: np.ndarray) -> np.ndarray:
        """
        Build adjacency from co-occurrence statistics (Equation 8)
        A^3_ij = O_ij / T_i
        """
        
        # Normalize by row sums (conditional probability)
        row_sums = cooccurrence_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        
        adj = cooccurrence_counts / row_sums
        self.adj_cooccur = adj
        return adj
    
    def binarize_adjacency(self, adj: np.ndarray, threshold: float) -> np.ndarray:
        """Binarize adjacency matrix using threshold (Equation 9)"""
        return (adj >= threshold).astype(np.float32)
    
    def get_all_adjacencies(self, threshold: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """Get all three binarized and normalized adjacency matrices"""
        
        adj_1 = self.binarize_adjacency(self.adj_fskg, threshold)
        adj_2 = self.binarize_adjacency(self.adj_hyperclass, threshold)
        adj_3 = self.binarize_adjacency(self.adj_cooccur, threshold)
        
        # Normalize each
        adj_1_norm = self._normalize_adjacency(adj_1)
        adj_2_norm = self._normalize_adjacency(adj_2)
        adj_3_norm = self._normalize_adjacency(adj_3)
        
        return (torch.FloatTensor(adj_1_norm),
                torch.FloatTensor(adj_2_norm),
                torch.FloatTensor(adj_3_norm))
    
    def _normalize_adjacency(self, adj: np.ndarray) -> np.ndarray:
        """Normalize adjacency matrix"""
        adj = adj + np.eye(adj.shape[0])
        degrees = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
