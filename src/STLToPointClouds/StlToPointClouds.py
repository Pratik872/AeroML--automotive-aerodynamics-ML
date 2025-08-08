import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class STLToPointCloudConverter:
    """Convert STL meshes to point clouds for PointNet"""

    def __init__(self, num_points = 2048):
        self.num_points = num_points

    def load_stl_mesh(self, stl_path):
        """Load STL file using trimesh"""

        loaded = trimesh.load(str(stl_path))

        #Handle Scene Objects
        if isinstance(loaded, trimesh.Scene):
            mesh = list(loaded.geometry.values())[0]

        else:
            mesh = loaded

        return mesh
    

    def mesh_to_pointcloud(self, mesh):
        """Convert mesh to point cloud with normals"""

        #Sample points uniformly from mesh surface
        points, face_indices = mesh.sample(self.num_points, return_index = True)

        #Get surface normals at sample points
        face_normals = mesh.face_normals[face_indices]

        #Combine points and normals(x, y, z, nx, ny, nz)
        point_cloud = np.concatenate([points, face_normals], axis=1)

        return point_cloud.astype(np.float32)
    
    def convert_stl_to_pointcloud(self, stl_path):
        """Complete pipeline: STL → point cloud"""

        try:

            mesh = self.load_stl_mesh(stl_path)
            point_cloud = self.mesh_to_pointcloud(mesh)
            return point_cloud, True
        
        except Exception as e:
            print(f"Error converting {stl_path}: {e}")
            # Return zero point cloud if conversion fails
            return np.zeros((self.num_points, 6), dtype=np.float32), False
        
    def visualize_pointcloud(self, point_cloud, title="Point Cloud"):
        """Visualize point cloud (points only, not normals)"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract points (first 3 columns)
        points = point_cloud[:, :3]
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
class AeroDynamicsPointCloudDataset(Dataset):
    """Dataset for loading point clouds with drag coefficients"""

    def __init__(self, base_path = "data/drivaerml_data", num_points = 2048, subset_size = None):
        self.base_path = Path(base_path)
        self.num_points = num_points
        self.converter = STLToPointCloudConverter(num_points)

        #Load enhanced dataset with drag_coefficients
        enhanced_data_path = self.base_path / "enhanced_dataset.csv"
        self.data = pd.read_csv(enhanced_data_path)

        if subset_size:
            self.data = self.data.head(subset_size)

        print(f"Dataset loaded: {len(self.data)} vehicles")
        print(f"Point cloud size: {num_points} points × 6 features (xyz + normals)")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get point cloud and drag coefficient"""

        row = self.data.iloc[idx]
        run_id = int(row['run'])
        drag_coeff = float(row['Cd'])

        # Round Cd to 2 decimals for better training
        drag_coeff = round(drag_coeff, 2)


        # STL path
        stl_path = self.base_path / f"run_{run_id}" / f"drivaer_{run_id}.stl"

        # Convert to point cloud
        point_cloud, success = self.converter.convert_stl_to_pointcloud(stl_path)

        if not success:
            # Return zeros if conversion failed
            point_cloud = np.zeros((self.num_points, 6), dtype=np.float32)
        
        # Transpose for PointNet: (6, num_points)
        point_cloud = point_cloud.T
        
        return torch.from_numpy(point_cloud), torch.tensor(drag_coeff, dtype=torch.float32)