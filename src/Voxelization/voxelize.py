import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


#Cuda Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")  

class STLToVoxelConverter:
    """Convert STL meshes to voxel grids for 3D CNN processing"""
    def __init__(self, grid_size = 64, bounds_padding = 0.1):
        """
        Args:
            grid_size: Resolution of voxel grid (grid_size^3 total voxels)
            bounds_padding: Extra space around mesh (0.1 = 10% padding)
        """

        self.grid_size = grid_size
        self.bounds_padding = bounds_padding

    def load_stl_mesh(self, stl_path):
        """Load STL file using trimesh (your proven pattern from Phase 2)"""

        loaded = trimesh.load(str(stl_path))
        # Handle Scene objects (from your Phase 2 code)
        if isinstance(loaded, trimesh.Scene):
            mesh = list(loaded.geometry.values())[0]
        else:
            mesh = loaded
            
        return mesh
    
    def mesh_to_voxels(self, mesh):
        """Convert mesh to binary voxel grid"""
        
        # Get mesh bounds and make writable copy
        bounds = mesh.bounds.copy()
        mesh_size = bounds[1] - bounds[0]
        
        # Add padding
        padding = mesh_size * self.bounds_padding
        bounds[0] -= padding
        bounds[1] += padding
        
        # Create voxel grid coordinates
        x = np.linspace(bounds[0,0], bounds[1,0], self.grid_size)
        y = np.linspace(bounds[0,1], bounds[1,1], self.grid_size)
        z = np.linspace(bounds[0,2], bounds[1,2], self.grid_size)
        
        # Create 3D grid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # For non-watertight meshes (car surfaces), use proximity-based voxelization
        if not mesh.is_watertight:
            try:
                # Method 1: Try signed distance
                distances = mesh.nearest.signed_distance(points)
                voxel_size = mesh_size.max() / self.grid_size
                threshold = voxel_size * 0.7
                inside = np.abs(distances) <= threshold
            except:
                try:
                    # Method 2: Try direct distance calculation
                    closest_points, distances, _ = mesh.nearest.on_surface(points)
                    voxel_size = mesh_size.max() / self.grid_size
                    threshold = voxel_size * 0.7
                    inside = distances <= threshold
                except:
                    # Method 3: Fallback - use mesh voxelization
                    voxel_size = mesh_size.max() / self.grid_size
                    voxelized = mesh.voxelized(pitch=voxel_size)
                    # Resize to match our grid
                    from scipy.ndimage import zoom
                    target_shape = (self.grid_size, self.grid_size, self.grid_size)
                    inside = zoom(voxelized.matrix.astype(float), 
                                np.array(target_shape) / np.array(voxelized.matrix.shape), 
                                order=0) > 0.5
                    return inside.astype(np.float32)
        else:
            # Use contains method for closed meshes
            inside = mesh.contains(points)
        
        # Reshape to 3D grid
        voxel_grid = inside.reshape(self.grid_size, self.grid_size, self.grid_size)
        
        return voxel_grid.astype(np.float32)
    
    def convert_stl_to_voxels(self, stl_path):
        """Complete pipeline: STL file → voxel grid"""
        try:
            mesh = self.load_stl_mesh(stl_path=stl_path)
            voxels = self.mesh_to_voxels(mesh)

            return voxels, True
        
        except Exception as e:
            print(f"Error converting {stl_path}: {e}")
            return np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float32), False
        
    def visualize_voxels(self, voxels, title="Voxel Grid"):
        """Visualize voxel grid in 3D"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get coordinates of filled voxels
        x, y, z = np.where(voxels > 0.5)
        
        # Plot voxels
        ax.scatter(x, y, z, c='blue', alpha=0.6, s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.show()
        
        return fig

class AerodynamicsVoxelDataset(Dataset):
    """Dataset for loading voxelized vehicles with drag coefficients"""

    def __init__(self, base_path = './data/drivaerml_data', grid_size=64, subset_size=None):
        """
        Args:
            base_path: Path to drivaerml_data folder
            grid_size: Voxel grid resolution
            subset_size: If set, only use first N vehicles (for testing)
        """
        self.base_path = Path(base_path)
        self.grid_size = grid_size
        self.converter = STLToVoxelConverter(grid_size)

        # Load enhanced dataset with drag coefficients (from your Phase 2)
        enhanced_data_path = self.base_path / "enhanced_dataset.csv"
        self.data = pd.read_csv(enhanced_data_path)
        
        if subset_size:
            self.data = self.data.head(subset_size)
        
        print(f"Dataset loaded: {len(self.data)} vehicles")
        print(f"Drag coefficient range: {self.data['Cd'].min():.3f} - {self.data['Cd'].max():.3f}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get voxelized vehicle and drag coefficient"""
        row = self.data.iloc[idx]
        run_id = int(row['run'])
        drag_coeff = float(row['Cd'])

        # STL path (using your existing structure)
        stl_path = self.base_path / f"run_{run_id}" / f"drivaer_{run_id}.stl"

        #convert to voxels
        voxels, success = self.converter.convert_stl_to_voxels(stl_path)

        if not success:
             # Return zeros if conversion failed
            voxels = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)

        # Add channel dimension for CNN: (1, D, H, W)
        voxels = voxels[np.newaxis, ...]
        
        return torch.from_numpy(voxels), torch.tensor(drag_coeff, dtype=torch.float32)
