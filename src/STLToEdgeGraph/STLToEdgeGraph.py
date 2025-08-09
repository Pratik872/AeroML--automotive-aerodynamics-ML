import numpy as np
import trimesh
import torch
from collections import defaultdict
from torch_geometric.data import Data



class STLToEdgeGraphConverter:
    """
    Convert STL triangle mesh to edge-based graph for MeshCNN
    
    Technical Explanation:
    - Traditional approach: vertices as nodes
    - MeshCNN approach: EDGES as nodes
    - Why? Edges capture geometric transitions crucial for aerodynamics
    """

    def __init__(self):
        self.edge_features_dim = 4  #Start with 4 key features

    
    def mesh_to_edge_graph(self, mesh):
        """
        Convert triangle mesh to edge graph representation
        
        Process:
        1. Extract all edges from triangular faces
        2. Compute edge features (dihedral angles, lengths, etc.)
        3. Build edge-to-edge adjacency (edges sharing vertices)
        4. Return PyTorch Geometric Data object
        """

        try:
            #Clean mesh if needed
            if not mesh.is_watertight:
                mesh.process()

            vertices = mesh.vertices
            faces = mesh.faces


            #Step1: Extract edges and build mappings
            edges, edge_to_faces = self._extract_edges_from_faces(faces)

            #Step2: Compute edge features
            edge_features = self._compute_edge_features(vertices, edges, faces, edge_to_faces)

            #Step3: Build edge-to-edge adjacency
            edge_adjacency = self._build_edge_adjacency(edges)


            #Step4: Create Pytorch Geometric data
            x = torch.tensor(edge_features, dtype=torch.float32)
            edge_index = torch.tensor(edge_adjacency, dtype=torch.long).t().contiguous()
            
            return Data(x=x, edge_index=edge_index)


        except Exception as e:
            print(f"Error converting mesh to edge graph: {e}")
            # Return dummy graph for robustness
            return Data(x=torch.zeros((10, self.edge_features_dim)), 
                       edge_index=torch.zeros((2, 10), dtype=torch.long))
        

    def _extract_edges_from_faces(self, faces):
        """
        Extract unique edges from triangular faces
        
        Technical Detail:
        - Each triangle has 3 edges: (v0,v1), (v1,v2), (v2,v0)
        - We store edges as (min_vertex, max_vertex) to avoid duplicates
        - Track which faces share each edge (needed for dihedral angles)
        """

        edge_to_faces = defaultdict(list)

        for face_idx, face in enumerate(faces):
            v0, v1, v2 = face

            #Three edges per triangle

            edges_in_face = [
                (min(v0, v1), max(v0, v1)),
                (min(v1, v2), max(v1, v2)),
                (min(v2,v0), max(v2, v0))
            ]

            for edge in edges_in_face:
                edge_to_faces[edge].append(face_idx)

        #Convert to list for indexing
        edges = list(edge_to_faces.keys())

        return edges, edge_to_faces
    

    def _compute_edge_features(self, vertices, edges, faces, edge_to_faces):
        """
        Compute features for each edge
        
        Features Explained:
        1. Dihedral Angle: Angle between adjacent faces (MOST IMPORTANT for aerodynamics)
           - Sharp edges (high angle) → flow separation → higher drag
           - Smooth edges (low angle) → attached flow → lower drag
        
        2. Edge Length: Physical size of edge
           - Longer edges → larger influence region
           - Important for mesh resolution understanding
        
        3. Edge Type: Boundary vs Internal
           - Boundary edges (only 1 face) → vehicle outline
           - Internal edges (2 faces) → surface details
        
        4. Face Area Ratio: Size difference of adjacent faces
           - Large ratio → mesh resolution transition
           - Important for understanding local geometry scale
        """
         
        edge_features = []

        for edge in edges:
            v1_idx, v2_idx = edge
            v1, v2 = vertices[v1_idx], vertices[v2_idx]

            #Feature 1: Edge length
            edge_length = np.linalg.norm(v2 - v1)

            #Feature 2: Dihedral Angle
            face_indices = edge_to_faces[edge]
            if len(face_indices) == 2: #Internal edge with 2 adjacent faces
                dihedral_angle = self._compute_dihedral_angle(
                vertices, faces, face_indices[0], face_indices[1], edge
                )

                edge_type = 0.0
            else:  #Boundary edge has only 1 face
                dihedral_angle = np.pi
                edge_type = 1.0


            #Feature3: Face Area Ratio
            if len(face_indices) == 2:
                area1 = self._triangle_area(vertices[faces[face_indices[0]]])
                area2 = self._triangle_area(vertices[faces[face_indices[1]]])
                area_ratio = max(area1, area2) / (min(area1, area2) + 1e-8)
            else:
                area_ratio = 1.0

            #Normalize features
            features = [
                dihedral_angle / np.pi,
                np.log(edge_length + 1e-8),
                edge_type,
                np.log(area_ratio + 1e-8)
            ]

            edge_features.append(features)

        return np.array(edge_features, dtype=np.float32)
    
    def _compute_dihedral_angle(self, vertices, faces, face1_idx, face2_idx, shared_edge):
        """
        Compute dihedral angle between two faces sharing an edge
        
        Technical Explanation:
        - Dihedral angle = angle between face normal vectors
        - Sharp edges (high dihedral) → flow separation
        - This is THE most important feature for aerodynamics!
        """
        face1 = faces[face1_idx]
        face2 = faces[face2_idx]

        #Compute face normals
        normal1 = self._face_normal(vertices[face1])
        normal2 = self._face_normal(vertices[face2])

        #Compute angle between normals
        cos_angle = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return angle
    

    def _face_normal(self, face_vertices):
        """Compute unit normal vector of triangular face"""

        v0, v1, v2 = face_vertices
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        return normal / (norm + 1e-8)
    
    def _triangle_area(self, face_vertices):
        """Compute area of triangular face"""
        v0, v1, v2 = face_vertices
        edge1 = v1 - v0
        edge2 = v2 - v0
        return 0.5 * np.linalg.norm(np.cross(edge1, edge2))
    
    def _build_edge_adjacency(self, edges):
        """
        Build edge-to-edge adjacency matrix
        
        Technical Explanation:
        - Two edges are adjacent if they share a vertex
        - This preserves mesh connectivity in the graph
        - Critical for MeshCNN message passing
        """
        edge_adjacency = []
        
        # Create vertex-to-edges mapping
        vertex_to_edges = defaultdict(list)
        for edge_idx, (v1, v2) in enumerate(edges):
            vertex_to_edges[v1].append(edge_idx)
            vertex_to_edges[v2].append(edge_idx)
        
        # Build adjacency: edges sharing vertices are connected
        for edge_idx, (v1, v2) in enumerate(edges):
            adjacent_edges = set()
            adjacent_edges.update(vertex_to_edges[v1])
            adjacent_edges.update(vertex_to_edges[v2])
            adjacent_edges.remove(edge_idx)  # Remove self
            
            for adj_edge_idx in adjacent_edges:
                edge_adjacency.append([edge_idx, adj_edge_idx])
        
        return edge_adjacency if edge_adjacency else [[0, 0]]  # Avoid empty adjacency