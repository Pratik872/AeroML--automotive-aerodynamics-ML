import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')



class STLDataAnalysis:

    def __init__(self, file_path):
        self.stl_df = None
        self.file_path = file_path


    def load_stl_features(self):
        """Load STL features dataset."""
        self.stl_df = pd.read_csv(str(self.file_path))
        print(f"📊 STL Features Dataset:")
        print(f"   Samples: {len(self.stl_df)}")
        print(f"   Features: {len(self.stl_df.columns)-1}")  # Exclude run_id
        print(f"   Memory usage: {self.stl_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        return self.stl_df
    

    def basic_statistics(self):
        """Display basic statistics for STL features."""
        print(f"\n📈 Basic Statistics:")
        
        # Exclude run_id for analysis
        feature_cols = [col for col in self.stl_df.columns if col != 'run_id']
        
        # Basic stats
        stats_df = self.stl_df[feature_cols].describe()
        print(stats_df.round(4))
        
        # Missing values check
        missing_values = self.stl_df[feature_cols].isnull().sum()
        if missing_values.sum() > 0:
            print(f"\n⚠️  Missing values found:")
            print(missing_values[missing_values > 0])
        else:
            print(f"\n✅ No missing values")
        
        return stats_df

    def analyze_feature_distributions(self):
        """Analyze distribution of each feature."""
        print(f"\n📊 Feature Distribution Analysis:")
        
        feature_cols = [col for col in self.stl_df.columns if col != 'run_id']
        
        # Create subplots for distributions
        n_features = len(feature_cols)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten()
        
        distribution_summary = []
        
        for i, col in enumerate(feature_cols):
            data = self.stl_df[col]
            
            # Plot histogram with robust binning
            try:
                unique_vals = data.nunique()
                data_range = data.max() - data.min()
                
                if unique_vals <= 1 or data_range == 0:
                    # Constant feature - show as single bar
                    axes[i].bar([data.iloc[0]], [len(data)], alpha=0.7, color='orange')
                    axes[i].text(data.iloc[0], len(data)/2, f'Constant\n{data.iloc[0]:.3f}', 
                                ha='center', va='center', fontsize=8)
                    axes[i].set_title(f'{col} (CONSTANT)', fontsize=10, color='red')
                else:
                    # Variable feature - adaptive binning
                    n_bins = min(30, max(5, unique_vals))
                    axes[i].hist(data, bins=n_bins, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{col}', fontsize=10)
                    
                    # Add mean line
                    mean_val = data.mean()
                    axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, 
                                label=f'Mean: {mean_val:.3f}')
                    axes[i].legend(fontsize=8)
                    
            except Exception as e:
                # Fallback for any binning issues
                axes[i].text(0.5, 0.5, f'Plot Error\n{col}', ha='center', va='center', 
                            transform=axes[i].transAxes, fontsize=10)
                axes[i].set_title(f'{col} (ERROR)', fontsize=10, color='red')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            
            # Distribution analysis
            mean_val = data.mean()
            std_val = data.std()
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            distribution_summary.append({
                'feature': col,
                'mean': mean_val,
                'std': std_val,
                'min': data.min(),
                'max': data.max(),
                'range': data.max() - data.min(),
                'cv': std_val / mean_val if mean_val != 0 else np.inf,
                'skewness': skewness,
                'kurtosis': kurtosis
            })
        
        # Remove empty subplots
        for i in range(n_features, len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        plt.show()
        
        # Summary table
        dist_df = pd.DataFrame(distribution_summary)
        print(f"\n📋 Distribution Summary:")
        print(dist_df.round(4))
        
        return dist_df


    def analyze_realistic_values(self):
        """Check if extracted values are realistic for automotive applications."""
        print(f"\n🚗 Automotive Realism Check:")
        
        # Expected ranges for realistic sedans
        realistic_ranges = {
            'volume': (8, 20, 'm³'),
            'surface_area': (25, 45, 'm²'),
            'length': (3.5, 5.5, 'm'),
            'width': (1.5, 2.2, 'm'),
            'height': (1.2, 2.0, 'm'),
            'length_width_ratio': (2.0, 3.5, 'ratio'),
            'height_length_ratio': (0.25, 0.45, 'ratio'),
            'aspect_ratio': (2.0, 4.0, 'ratio'),
            'compactness': (0.1, 100, 'unitless'),
            'volume_efficiency': (0.05, 0.25, 'ratio')
        }
        
        realism_check = []
        
        for feature, (min_expected, max_expected, unit) in realistic_ranges.items():
            if feature in self.stl_df.columns:
                data = self.stl_df[feature]
                actual_min = data.min()
                actual_max = data.max()
                actual_mean = data.mean()
                
                # Check if values are in expected range
                in_range = (actual_min >= min_expected * 0.8) and (actual_max <= max_expected * 1.2)
                
                realism_check.append({
                    'feature': feature,
                    'expected_min': min_expected,
                    'expected_max': max_expected,
                    'actual_min': actual_min,
                    'actual_max': actual_max,
                    'actual_mean': actual_mean,
                    'unit': unit,
                    'realistic': '✅' if in_range else '❌'
                })
                
                print(f"   {feature:20s}: {actual_mean:8.3f} {unit:10s} [{actual_min:.3f} - {actual_max:.3f}] {realism_check[-1]['realistic']}")
        
        return pd.DataFrame(realism_check)
    


    def correlation_with_target(self, target_file_path):
        """Analyze correlation of STL features with drag coefficient."""
        print(f"\n🎯 Correlation with Drag Coefficient:")
        
        try:
            # Load target data
            target_df = pd.read_csv(target_file_path)
            
            # Merge with STL features
            merged = pd.merge(self.stl_df, target_df[['run', 'cd']], on='run', how='inner')
            
            # Calculate correlations
            feature_cols = [col for col in self.stl_df.columns if col != 'run']
            correlations = []
            
            for col in feature_cols:
                corr_coef = merged[col].corr(merged['cd'])
                correlations.append({
                    'feature': col,
                    'correlation': corr_coef,
                    'abs_correlation': abs(corr_coef)
                })
            
            # Sort by absolute correlation
            corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
            
            print(f"   Top 10 most correlated STL features:")
            for i, row in corr_df.head(10).iterrows():
                print(f"   {row['feature']:25s}: {row['correlation']:6.3f}")
            
            # Visualize top correlations
            plt.figure(figsize=(12, 8))
            top_features = corr_df.head(15)
            
            colors = ['red' if x < 0 else 'blue' for x in top_features['correlation']]
            plt.barh(range(len(top_features)), top_features['correlation'], color=colors, alpha=0.7)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Correlation with Drag Coefficient')
            plt.title('Top 15 STL Features - Correlation with Drag Coefficient')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            return corr_df
            
        except FileNotFoundError:
            print("   ❌ Target file not found. Skipping correlation analysis.")
            return None



    def feature_variability_analysis(self):
        """Analyze feature variability for ML usefulness."""
        print(f"\n📏 Feature Variability Analysis:")
        
        feature_cols = [col for col in self.stl_df.columns if col != 'run_id']
        variability = []
        
        for col in feature_cols:
            data = self.stl_df[col]
            cv = data.std() / data.mean() if data.mean() != 0 else np.inf
            range_norm = (data.max() - data.min()) / data.mean() if data.mean() != 0 else np.inf
            
            variability.append({
                'feature': col,
                'coefficient_of_variation': cv,
                'normalized_range': range_norm,
                'std_dev': data.std(),
                'usefulness': 'High' if cv > 0.1 else 'Medium' if cv > 0.05 else 'Low'
            })
        
        var_df = pd.DataFrame(variability).sort_values('coefficient_of_variation', ascending=False)
        
        print(f"   Features ranked by variability (CV = std/mean):")
        for i, row in var_df.head(10).iterrows():
            print(f"   {row['feature']:25s}: CV={row['coefficient_of_variation']:6.3f} ({row['usefulness']})")
        
        return var_df
    
