# Automotive Aerodynamics using Machine Learning - Drag Coefficient Prediction

## Overview
Predicting Automotive Drag coefficients from geometric features and 3D mesh data.

**Dataset:** 484 DrivAer vehicle geometries with validated CFD drag coefficients  


## Project Evolution

### Phase 1: Traditional ML ✅ (Baseline)
**Approach:** 16 geometric parameters → ML models  
**Best Result:** XGBoost 87.9% R²

| Model | R² Score |
|-------|----------|
| Linear Regression | 87.3% |
| Ridge Regression | 87.4% |
| Random Forest | 78.7% |
| **XGBoost** | **87.9%** |

**Limitation:** Global features missed local aerodynamic details

### Phase 2: STL Feature Engineering ✅
**Approach:** Extracted 20 3D features (volume, curvature, surface area)  
**Result:** 87.5% R² (marginal improvement)  
**Finding:** Traditional ML reached performance plateau

### Phase 3: 3D CNNs
**Approach:** STL → voxelization (64³ grid) → 3D CNN

**Architecture:**
```
Input: 64×64×64 voxels (262,144 voxels)
    ↓
Conv3D(1→32) + MaxPool3D
    ↓
Conv3D(32→64) + MaxPool3D  
    ↓
Conv3D(64→128) + MaxPool3D
    ↓
Flatten → FC(131,072→1024→512→1)
```
**Parameters:** 4.5M  

### Phase 4: PointNet
**Approach:** STL → point clouds (2048 points + normals) → PointNet

**Architecture:**
```
Input: 2048×6 (xyz + normals)
    ↓
T-Net (Transform) → SharedMLP(64,128,1024)
    ↓
MaxPool → FC(1024→512→256→1)
```
**Parameters:** 400K → 80K (reduced)  


### Phase 5: Graph Neural Networks- MeshCNN
**Approach:** STL → edge graphs → Graph Neural Network  
**Key Insight:** Edges capture geometric transitions crucial for aerodynamics

## Methodology

### 1. Traditional ML Models
**Approach:** Feature extraction from 16 geometric parameters (length, width, height, angles, etc.)

**Models Implemented:**
- Linear Regression: Direct linear mapping
- Ridge Regression: L2 regularization  
- Random Forest: Ensemble of decision trees
- XGBoost: Gradient boosting

**Best Performance:** XGBoost 87.9% R²

### 2. 3D Convolutional Neural Networks
**Data Processing:**
- STL mesh → voxelization (64³ grid)
- 262,144 voxels per vehicle (mostly empty space)

**Architecture:**
```
Input: 64×64×64×1 voxels
Conv3D(1→32, kernel=3) + MaxPool3D
Conv3D(32→64, kernel=3) + MaxPool3D  
Conv3D(64→128, kernel=3) + MaxPool3D
Flatten(16,384) → Dense(1024) → Dense(512) → Dense(1)
```
**Parameters:** 4.5M  


### 3. PointNet
**Data Processing:**
- STL mesh → point cloud sampling (2048 points)
- Features: xyz coordinates + surface normals (6D)

**Architecture:**
```
Input: 2048×6 points
T-Net (transform) → SharedMLP(64,128,1024)
MaxPooling(2048→1) → FC(1024→512→256→1)
```
**Parameters:** 400K → 80K (reduced)  


### 4. MeshCNN (Current Approach)
**Data Processing:**
- STL mesh → edge graph (no information loss)
- Edge features: dihedral angles, lengths, boundary types, area ratios

**Edge Graph Conversion:**
```
Triangle Mesh → Edge Extraction → Feature Computation → Graph Construction
```

**Edge Features (4D per edge):**
- **Dihedral Angle:** Angle between adjacent faces (0°-180°)
- **Edge Length:** Physical edge size (log scale) 
- **Edge Type:** Boundary vs Internal classification
- **Face Area Ratio:** Adjacent face size difference

**MeshCNN Architecture:**
```
Input: Edge Graph (7,759 edges × 4 features)
    ↓
EdgeConv Layer 1: 4 → 64 features
(Message passing between adjacent edges)
    ↓ ReLU + BatchNorm
EdgeConv Layer 2: 64 → 64 features
(Spatial pattern learning)  
    ↓ ReLU + BatchNorm
Global Mean Pool: 7,759×64 → 1×64
    ↓
Regression Head: 64 → 32 → 1
```
**Parameters:** 11,201

**Edge Convolution Operation:**
```python
def EdgeConv(x_self, x_neighbors):
    combined = concat([x_self, x_neighbors], dim=1)
    return Linear(combined)  # Learn local geometric patterns
```

## Data Preprocessing

### Dataset Characteristics
**DrivAer Dataset:**
- 484 complete sedan geometries
- STL files: 2K-5K vertices each, non-watertight meshes  
- Drag coefficients: 0.237-0.341 (CFD validated)
- Split: 388 train, 96 validation (20%)

**File Structure:**
```
drivaerml_data/
├── run_1/drivaer_1.stl
├── run_2/drivaer_2.stl
├── ...
├── run_484/drivaer_484.stl
└── phase2_ready/enhanced_dataset.csv
```

### Feature Engineering
**Edge Feature Computation:**
1. **Dihedral Angle Calculation:**
   - Compute face normals for adjacent triangles
   - Calculate angle between normal vectors
   - Normalize to [0,1] range

2. **Edge Length:** Log-scaled physical distance
3. **Boundary Detection:** Edges shared by 1 vs 2 faces  
4. **Area Ratio:** `max(area1, area2) / min(area1, area2)`


## Key Technical Innovations

### 1. Edge-Based Representation
- **Traditional GNNs:** Vertices as nodes → miss geometric transitions
- **MeshCNN:** Edges as nodes → directly capture surface curvature changes

### 2. Aerodynamic Relevance  
- **Dihedral angles:** Sharp edges (high angles) → flow separation → higher drag
- **Edge connectivity:** Preserves spatial relationships between surface features
- **No information loss:** Unlike voxelization/point sampling

### 3. Message Passing
- Each edge learns from neighboring edges
- Captures patterns: "sharp edge + smooth neighbors = transition zone"
- Preserves mesh topology through graph structure

## Results

![XGB](https://github.com/Pratik872/AeroML--automotive-aerodynamics-ML/blob/main/readme_resources/XGBoost.png)

![Ridge_Lasso](https://github.com/Pratik872/AeroML--automotive-aerodynamics-ML/blob/main/readme_resources/Ridge_Lasso.png)

![3DCNN](https://github.com/Pratik872/AeroML--automotive-aerodynamics-ML/blob/main/readme_resources/3DCNN_losses.png)

![PointNet](https://github.com/Pratik872/AeroML--automotive-aerodynamics-ML/blob/main/readme_resources/PointNet_losses.png)

![MeshCNN](https://github.com/Pratik872/AeroML--automotive-aerodynamics-ML/blob/main/readme_resources/MeshCNN.png)


## Future Work

**484 samples insufficient:**
- DrivAerNet++: 8,000 vehicles (84GB dataset)
- Hybrid approach: Traditional ML + edge features
- Transfer learning from larger datasets

**Architecture improvements:**
- Edge pooling for mesh simplification
- Attention mechanisms for important edges
- Multi-scale feature extraction

## Built with 🛠️
- Python 3.12.0
- PyTorch + PyTorch Geometric
- Libraries: trimesh, pandas, scikit-learn, matplotlib
- GPU: CUDA support
- Development: Jupyter Notebook

## References
- [MeshCNN Paper](https://arxiv.org/abs/1809.05910)
- [DrivAer Dataset](https://www.aer.mw.tum.de/en/research-groups/automotive/drivaer/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

---

**Status:** Model currently training. Results pending...

**Hypothesis:** Edge-based geometric representation will capture aerodynamic patterns missed by previous approaches, achieving breakthrough >90% R² accuracy.