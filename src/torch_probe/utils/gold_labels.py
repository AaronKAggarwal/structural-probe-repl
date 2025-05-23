# src/torch_probe/utils/gold_labels.py
from typing import List, Dict, Set
import numpy as np
from collections import deque

def _build_adjacency_list(num_tokens: int, head_indices: List[int]) -> Dict[int, List[int]]:
    """Builds an undirected adjacency list from head indices."""
    adj: Dict[int, List[int]] = {i: [] for i in range(num_tokens)}
    root_token_indices = []
    for i, head in enumerate(head_indices):
        if head == -1: # Token i is a root
            root_token_indices.append(i)
            continue
        if head < 0 or head >= num_tokens:
            # This case should ideally be caught earlier or indicate a parsing error
            # For robustness, we might treat it as a disconnected node or skip the edge
            # print(f"Warning: Invalid head index {head} for token {i}. Skipping edge.")
            continue
        if i == head: # Self-loop, usually means root if not -1
            root_token_indices.append(i)
            continue
        
        adj[i].append(head)
        adj[head].append(i)
    
    # Handle forests: if multiple roots, connect them to a virtual super-root (not done here, assumes tree)
    # For single tree, this adj list is fine.
    return adj

def calculate_tree_depths(head_indices: List[int]) -> List[int]:
    """
    Calculates the depth of each token in a dependency tree.
    Depth of root is 0.

    Args:
        head_indices: List of 0-indexed head indices. Root token's head is -1
                      or points to itself (if multiple roots, first one found is used).

    Returns:
        List of depths for each token.
    """
    num_tokens = len(head_indices)
    if num_tokens == 0:
        return []

    depths = [-1] * num_tokens
    adj = _build_adjacency_list(num_tokens, head_indices)
    
    # Find root(s) - first token with head == -1 or head == self_index
    root_idx = -1
    for i, head in enumerate(head_indices):
        if head == -1 or head == i:
            root_idx = i
            break
    
    if root_idx == -1 and num_tokens > 0: # Should not happen in valid CoNLLU tree
        # Fallback: assume first token is root if no explicit root found
        # print("Warning: No explicit root found, assuming token 0 as root for depth calculation.")
        root_idx = 0 

    if root_idx != -1:
        q = deque([(root_idx, 0)])
        visited: Set[int] = {root_idx}
        depths[root_idx] = 0
        
        while q:
            curr, d = q.popleft()
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    depths[neighbor] = d + 1
                    q.append((neighbor, d + 1))
    
    # Handle disconnected components if any node was not visited (depth remains -1)
    # For this probe, we assume a connected tree.
    return depths


def calculate_tree_distances(head_indices: List[int]) -> np.ndarray:
    """
    Calculates pairwise shortest path distances between all tokens in a dependency tree.

    Args:
        head_indices: List of 0-indexed head indices. Root token's head is -1.

    Returns:
        A NumPy array of shape (num_tokens, num_tokens) with distances.
    """
    num_tokens = len(head_indices)
    if num_tokens == 0:
        return np.empty((0,0), dtype=int)

    distances = np.full((num_tokens, num_tokens), -1, dtype=int) # -1 for unreachable
    adj = _build_adjacency_list(num_tokens, head_indices)

    for start_node in range(num_tokens):
        q = deque([(start_node, 0)])
        visited: Set[int] = {start_node}
        distances[start_node, start_node] = 0
        
        head = 0 # For BFS queue
        bfs_q = [(start_node, 0)]
        visited_bfs = {start_node}
        
        while head < len(bfs_q):
            curr, d = bfs_q[head]
            head+=1
            
            for neighbor in adj.get(curr, []):
                if neighbor not in visited_bfs:
                    visited_bfs.add(neighbor)
                    distances[start_node, neighbor] = d + 1
                    bfs_q.append((neighbor, d + 1))
    
    # Check for unreachable nodes (should not happen in a tree if connected)
    if np.any(distances == -1):
        # print("Warning: Some nodes might be unreachable in distance calculation.")
        pass # Or handle as error depending on strictness for tree structure

    return distances

if __name__ == '__main__':
    # Example Usage
    # Sentence 1: Root is token 3 ("test"), 0-indexed.
    # 1 This (nsubj of test) -> head 3
    # 2 is (cop of test) -> head 3
    # 3 a (det of test) -> head 3
    # 4 test (root) -> head -1 (was 0 in CoNLLU)
    # 5 . (punct of test) -> head 3
    # Heads (0-indexed): [3, 3, 3, -1, 3]
    heads1 = [3, 3, 3, -1, 3] 
    print(f"Sentence 1 Heads: {heads1}")
    depths1 = calculate_tree_depths(heads1)
    print(f"Depths1: {depths1}") # Expected: [1, 1, 1, 0, 1]
    distances1 = calculate_tree_distances(heads1)
    print(f"Distances1:\n{distances1}")
    # Expected:
    # [[0 2 2 1 2]
    #  [2 0 2 1 2]
    #  [2 2 0 1 2]
    #  [1 1 1 0 1]
    #  [2 2 2 1 0]]

    # Sentence 2: Root token 0 ("Root"), then chain 0->1, 1->2
    # Heads (0-indexed): [-1, 0, 1]
    heads2 = [-1, 0, 1]
    print(f"\nSentence 2 Heads: {heads2}")
    depths2 = calculate_tree_depths(heads2)
    print(f"Depths2: {depths2}") # Expected: [0, 1, 2]
    distances2 = calculate_tree_distances(heads2)
    print(f"Distances2:\n{distances2}")
    # Expected:
    # [[0 1 2]
    #  [1 0 1]
    #  [2 1 0]]
    pass