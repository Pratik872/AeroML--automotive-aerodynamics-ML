import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import trimesh

class AnalyseData:

    def __init__(self, data_path):

        self.data_path = Path(data_path)
        self.run_dirs = sorted([d for d in self.data_path.iterdir() if d.is_dir() and d.name.startswith('run_')])
        self.forces_df = None
        self.geo_df = None

        print(f"Total Vehicles downloaded: {len(self.run_dirs)}")
        print(f"Data Directory: {self.data_path.absolute()}\n")

    def clean_drivaerml_data(self, forces_df, geo_df):
        """Clean and sync geometry/forces data for consistent ML dataset"""
        
        # Find missing runs
        forces_runs = set(forces_df['run'])
        geo_runs = set(geo_df['Run'])
        
        missing_forces = geo_runs - forces_runs
        missing_geo = forces_runs - geo_runs
        
        print(f"\nMissing CFD data for runs: {sorted(missing_forces)}")
        print(f"Missing geometry for runs: {sorted(missing_geo)}")
        
        # Keep only runs with both geometry and forces data
        valid_runs = forces_runs.intersection(geo_runs)
        print(f"\nValid runs with both data: {len(valid_runs)}")
        
        # Filter datasets
        forces_clean = forces_df[forces_df['run'].isin(valid_runs)].copy()
        geo_clean = geo_df[geo_df['Run'].isin(valid_runs)].copy()
        
        # Sort by run number for consistency
        forces_clean = forces_clean.sort_values('run').reset_index(drop=True)
        geo_clean = geo_clean.sort_values('Run').reset_index(drop=True)
        
        # Verify alignment
        assert len(forces_clean) == len(geo_clean), "Misaligned data after cleaning"
        assert list(forces_clean['run']) == list(geo_clean['Run']), "Run IDs don't match"
        
        print(f"\nCleaned data:")
        print(f"Forces: {len(forces_clean)} records")
        print(f"Geometry: {len(geo_clean)} records")
        print(f"Drag coefficient range: {forces_clean['cd'].min():.3f} - {forces_clean['cd'].max():.3f}")
        
        return forces_clean, geo_clean, valid_runs

    def explore_drivaerml_data(self):
        """Explore DrivAerML dataset structure and key metrics"""
        
        print("=== DrivAerML Dataset Exploration ===\n")

        #Load Summary Data
        force_all_file = self.data_path / "force_mom_all.csv"
        geo_params_file = self.data_path / "geo_parameters_all.csv"


        if force_all_file.exists():
            print("Loading force_mom_all.csv...")
            forces_df = pd.read_csv(force_all_file)
            # forces_df.drop(['Run'], axis=1, inplace=True)
            print(f"Shape: {forces_df.shape}")
            print(f"Columns: {list(forces_df.columns)}")
            print("\nFirst few rows: ")
            print(forces_df.head())
            print(forces_df.tail())

            #Drag Coeffiecient Analysis
            if "cd" in forces_df.columns:
                print(f"\n=== Drag Coefficient Analysis ===")
                print(f"Mean Cd: {forces_df['cd'].mean():.4f}")
                print(f"Std Cd: {forces_df['cd'].std():.4f}")
                print(f"Min Cd: {forces_df['cd'].min():.4f}")
                print(f"Max Cd: {forces_df['cd'].min():4f}")

                #Plot distribution
                plt.figure(figsize=(10,6))
                plt.subplot(1,2,1)
                plt.hist(forces_df['cd'], bins = 20, alpha = 0.7, edgecolor = 'black')
                plt.xlabel('Drag Coefficient(Cd)')
                plt.ylabel('Frequency')
                plt.title('Drag Coefficient Distribution')

                plt.subplot(1,2,2)
                plt.boxplot(forces_df['cd'])
                plt.ylabel('Drag Coeffiecient (Cd)')
                plt.title('Cd Box Plot')

                plt.tight_layout()
                plt.show()

        if geo_params_file.exists():
            print("\n=== Geometric Parameters ===")
            geo_df = pd.read_csv(geo_params_file)
            print(f"Shape: {geo_df.shape}")
            print(f"Columns: {list(geo_df.columns)}")
            print("\nFirst few rows:")
            print(geo_df.head())

        forces_df, geo_df, valid_runs = self.clean_drivaerml_data(forces_df, geo_df)
        print("Cataset Cleaning Complete...")
        print("Shapes of datasets after cleaning: ")
        print(f"Forces Dataset: {forces_df.shape}")
        print(f"Geometry Dataset: {geo_df.shape}")

        self.forces_df = forces_df
        self.geo_df = geo_df

        # Examine individual vehicle files
        print(f"\n=== Individual Vehicle Files ===")
        sample_run = self.run_dirs[0] if self.run_dirs else None

        if sample_run:
            print(f"Examining {sample_run.name}:")

            #STL File
            stl_files = list(sample_run.glob("*.stl"))
            if stl_files:
                stl_file = stl_files[0]
                stl_size = stl_file.stat().st_size / (1024*1024)  # MB
                print(f"  STL file: {stl_file.name} ({stl_size:.1f} MB)")

            # CSV file
            csv_files = list(sample_run.glob("*.csv"))
            if csv_files:
                csv_file = csv_files[0]
                sample_df = pd.read_csv(csv_file)
                print(f"  CSV file: {csv_file.name}")
                print(f"    Columns: {list(sample_df.columns)}")
                print(f"    Data:")
                print(sample_df)

        # 4. Dataset summary
        print(f"\n=== Dataset Summary ===")
        total_stl = len(list(self.data_path.glob("**/*.stl")))
        total_csv = len(list(self.data_path.glob("**/force_mom_*.csv")))
        
        total_size = sum(f.stat().st_size for f in self.data_path.glob("**/*") if f.is_file())
        total_size_mb = total_size / (1024*1024)
        
        print(f"STL files: {total_stl}")
        print(f"CSV files: {total_csv}")
        print(f"Total size: {total_size_mb:.1f} MB")
        
        return forces_df if force_all_file.exists() else None, geo_df if geo_params_file.exists() else None
    

    def load_vehicle_geometry(self, vehicle_id):
        """Load specific vehicle's STL geometry"""
        try:
            stl_path = Path(self.data_dir) / f"run_{vehicle_id}" / f"drivaer_{vehicle_id}.stl"
            mesh = trimesh.load(stl_path)
            return mesh
        
        except ImportError:
            print("Install trimesh: pip install trimesh")
            return None
