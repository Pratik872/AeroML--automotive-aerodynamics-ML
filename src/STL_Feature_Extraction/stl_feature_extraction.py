import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm


class STLFeatureExtraction:

    def __init__(self, base_path):
        
        self.base_path = base_path
        


    ## Mesh Visualization Functions

    def verify_stl_files(self):
        """Verify STL files exist and get basic statistics"""
        path = Path(self.base_path)

        #Load the valid run IDs from Phase1
        try:
            merged_data = pd.read_csv(path / "artifacts" / "ml_ready" /"merged_dataset.csv")
            valid_runs = merged_data['run'].tolist()
            print(f"Found {len(valid_runs)} valid runs from Phase 1")

        except FileNotFoundError:
            print("❌ merged_dataset.csv not found. Creating range 1-484...")
            valid_runs = list(range(1,485))

        #Check STL file existence
        existing_files = []
        missing_files = []
        file_sizes = []

        for run_id in valid_runs:
            stl_path = path / "drivaerml_data" /f"run_{run_id}" / f"drivaer_{run_id}.stl"
            if stl_path.exists():
                existing_files.append(run_id)
                file_sizes.append(stl_path.stat().st_size / (1024*1024)) #Size in MB

            else:
                missing_files.append(run_id)
        # Statistics
        print(f"\n📊 STL File Statistics:")
        print(f"✅ Existing files: {len(existing_files)}")
        print(f"❌ Missing files: {len(missing_files)}")
        if file_sizes:
            print(f"📦 Average file size: {np.mean(file_sizes):.1f} MB")
            print(f"📦 Size range: {min(file_sizes):.1f} - {max(file_sizes):.1f} MB")
        
        if missing_files:
            print(f"\n⚠️  Missing files: {missing_files[:10]}..." if len(missing_files) > 10 else f"\n⚠️  Missing files: {missing_files}")
        
        return existing_files, missing_files
        

    def visualize_with_open3d(self, existing_files, n_samples=2):
        """Visualize STL meshes using Open3D's interactive viewer."""
        
        for i, run_id in enumerate(existing_files[:n_samples]):
            stl_path = Path(self.base_path) / "drivaerml_data" / f"run_{run_id}" / f"drivaer_{run_id}.stl"
            
            # Load mesh
            mesh = o3d.io.read_triangle_mesh(str(stl_path))
            mesh.compute_vertex_normals()
            
            # Color the mesh
            mesh.paint_uniform_color([0.7, 0.7, 1.0])  # Light blue
            
            print(f"Opening vehicle run_{run_id} in 3D viewer...")
            print("Close the window to see the next vehicle")
            
            # Show interactive 3D visualization
            o3d.visualization.draw_geometries([mesh], 
                                            window_name=f"Vehicle run_{run_id}",
                                            width=800, height=600)
            

    ## Feature Extraction functions
    def extract_basic_features(self, mesh, scale_factor = 4.2):
        """Extract fundamental geometric properties with proper scaling."""
        features = {}

        #Basic properties(scaled to real units)
        features['volume'] =  abs(mesh.volume) * (scale_factor ** 3) #m3
        features['surface_area'] = mesh.area * (scale_factor ** 2) #m2

        #Bounding box dimensions
        bounds = mesh.bounds
        dimensions = (bounds[1] - bounds[0]) * scale_factor
        features['length'] = dimensions[0]
        features['width'] = dimensions[1]
        features['height'] = dimensions[2]


        #Derived ratios
        features['length_width_ratio'] = features['length'] / features['width']
        features['height_length_ratio'] = features['height'] / features['length']
        features['aspect_ratio'] = features['length'] / features['height']

        #Compactness (sphere = 1.0, more complex shapes < 1.0)
        features['compactness'] = (36 * np.pi * features['volume']**2) / features['surface_area']**3

        #Volume Ratios
        bbox_volume = np.prod(dimensions)
        features['volume_efficiency'] = features['volume'] / bbox_volume

        return features    
    

    def extract_curvature_features(self, mesh_o3d, scale_factor=4.2):
        """Extract surface curvature statistics using open3d."""
        features = {}

        try:
            #Scale mesh for proper curvature calculation
            mesh_scaled = mesh_o3d.scale(scale_factor, center=mesh_o3d.get_center())
            mesh_scaled.compute_vertex_normals()

            #Convert to point cloud for curvature analysis
            pcd = mesh_scaled.sample_points_uniformly(number_of_points = 1000)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

            # Simple curvature approximation using normal variation
            normals = np.asarray(pcd.normals)

            # Build KD tree for nearest neighbors
            pcd_tree = o3d.geometry.KDTreeFlann(pcd) 

            curvatures = []
            for i in range(len(pcd.points)):
                [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 10)
                if len(idx) > 1:
                    local_normals = normals[idx]
                    # Curvature approximated by normal variation
                    curvature = np.var(local_normals, axis=0).sum()
                    curvatures.append(curvature)
            
            curvatures = np.array(curvatures)
            
            features['mean_curvature'] = np.mean(curvatures)
            features['curvature_variance'] = np.var(curvatures)
            features['max_curvature'] = np.max(curvatures)

            return features

        except Exception as e:
            # Fallback values if curvature calculation fails
            features['mean_curvature'] = 0.1
            features['curvature_variance'] = 0.01
            features['max_curvature'] = 0.5
            print(f"Curvature calculation failed: {e}")


    def extract_mesh_complexity_features(self, mesh, scale_factor=4.2):
        """Extract mesh quality and complexity metrics."""
        features = {}

        features['vertex_count'] = len(mesh.vertices)
        features['face_count'] = len(mesh.faces)
        features['edge_count'] = len(mesh.edges)

        ##Mesh Density(per m2)
        scaled_area = mesh.area ** (scale_factor**2)
        features['vertex_density'] = features['vertex_count'] / scaled_area
        features['face_density'] = features['face_count'] / scaled_area

        #Edge length statistics(scaled to meters)
        edge_lengths = np.linalg.norm(mesh.vertices[mesh.edges[:, 1]] - mesh.vertices[mesh.edges[:, 0]], axis=1)
        edge_lengths_scaled = edge_lengths * scale_factor
        features['mean_edge_length'] = np.mean(edge_lengths_scaled)
        features['edge_length_variance'] = np.var(edge_lengths_scaled)

        return features
    


    def extract_stl_features(self, run_id, scale_factor=4.2):
        """Extract all features from a single STL file."""
        try:
            #Load with Trimesh
            path = Path(self.base_path) / "drivaerml_data" / f"run_{run_id}" / f"drivaer_{run_id}.stl"
            loaded = trimesh.load(str(path))

            if isinstance(loaded, trimesh.Scene):
                mesh = list(loaded.geometry.values())[0]
            else:
                mesh = loaded

            # Load with open3d for advanced processing
            mesh_o3d = o3d.io.read_triangle_mesh(str(path))

            # Extract different feature groups
            basic_features = self.extract_basic_features(mesh, scale_factor)
            
            curvature_features = self.extract_curvature_features(mesh_o3d, scale_factor)
            
            complexity_features = self.extract_mesh_complexity_features(mesh, scale_factor)
            

            #Combine all features
            all_features = {**basic_features, **curvature_features, **complexity_features}
            all_features['run_id'] = run_id

            return all_features

        except Exception as e:
            print(f"Failed to process run_{run_id}: {e}")
            return None 
        
    
    def process_all_stl_files(self, existing_files, max_files=None):
        """Extract features from all STL files."""
        files_to_process = existing_files[:max_files] if max_files else existing_files
        print(f"🚀 Processing {len(files_to_process)} STL files...")
        
        all_features = []
        failed_files = []
        
        # Process with progress bar
        for run_id in tqdm(files_to_process, desc="Extracting features"):
            features = self.extract_stl_features(run_id)
            if features:
                all_features.append(features)
            else:
                failed_files.append(run_id)
        
        # Convert to DataFrame
        stl_features_df = pd.DataFrame(all_features)
        
        print(f"✅ Successfully processed: {len(all_features)}")
        print(f"❌ Failed files: {len(failed_files)}")
        if len(stl_features_df.columns) > 1:
            print(f"📊 Extracted features: {len(stl_features_df.columns)-1}")
            print(f"📋 Feature names: {list(stl_features_df.columns[:-1])}")
        
        return stl_features_df, failed_files
    

    def save_features(stl_features_df, output_path="./data/artifacts/ml_ready/stl_features.csv"):
        """Save extracted features to CSV."""
        
        stl_features_df.to_csv(output_path, index=False)
        print(f"💾 Features saved to: {output_path}")