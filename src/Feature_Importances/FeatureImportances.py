import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path



class FeatureImportances:

    def __init__(self):
        self.merged_df = None
        self.importance_df = None

    def load_and_merge_data(self, stl_features_path, phase1_data_path):
        print("Loading and merging datasets...")

        #Load datasets
        stl_features = pd.read_csv(str(stl_features_path))
        phase1_data = pd.read_csv(str(phase1_data_path))


        #Merge
        self.merged_data = pd.merge(phase1_data, stl_features, on='run', how='inner')

        print(f"✅ Phase 1 features: {len(phase1_data.columns)-2}")  # Exclude run_id, cd
        print(f"✅ STL features: {len(stl_features.columns)-1}")     # Exclude run_id
        print(f"✅ Total features: {len(self.merged_data.columns)-2}")   # Exclude run_id, cd
        print(f"✅ Total samples: {len(self.merged_data)}")
        
        return self.merged_data
    


    def analyze_feature_correlations(self, threshold = 0.7):
        """Identify highly correlated features."""
        print(f"\n🔍 Analyzing feature correlations (threshold: {threshold})...")

        #Get feature columns
        feature_cols = [col for col in self.merged_data.columns if col not in ['run', 'cd']]
        features_df = self.merged_data[feature_cols]

        #Calculate correlation matrix
        corr_matrix = features_df.corr()

        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            print(f"⚠️  Found {len(high_corr_pairs)} highly correlated pairs:")
            for pair in high_corr_pairs:
                print(f"   {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        else:
            print(f"✅ No highly correlated pairs found")
        
        return corr_matrix, high_corr_pairs
    

    def calculate_feature_importance(self):
        """Calculate feature importance using multiple methods."""
        
        # Prepare data
        feature_cols = [col for col in self.merged_data.columns if col not in ['run_id', 'cd']]
        X = self.merged_data[feature_cols]
        y = self.merged_data['cd']
        
        # Method 1: Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': feature_cols,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        
        # Method 2: Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importance = pd.DataFrame({
            'feature': feature_cols,
            'mi_importance': mi_scores
        }).sort_values('mi_importance', ascending=False)
        
        # Method 3: Correlation with target
        target_corr = []
        for col in feature_cols:
            corr, _ = pearsonr(self.merged_data[col], y)
            target_corr.append(abs(corr))
        
        corr_importance = pd.DataFrame({
            'feature': feature_cols,
            'target_correlation': target_corr
        }).sort_values('target_correlation', ascending=False)
        
        # Combine all importance metrics
        importance_combined = rf_importance.merge(mi_importance, on='feature')\
                                        .merge(corr_importance, on='feature')
        
        # Calculate average rank
        importance_combined['rf_rank'] = importance_combined['rf_importance'].rank(ascending=False)
        importance_combined['mi_rank'] = importance_combined['mi_importance'].rank(ascending=False)
        importance_combined['corr_rank'] = importance_combined['target_correlation'].rank(ascending=False)
        importance_combined['avg_rank'] = (importance_combined['rf_rank'] + 
                                        importance_combined['mi_rank'] + 
                                        importance_combined['corr_rank']) / 3
        
        self.importance_df = importance_combined.sort_values('avg_rank')
        
        return self.importance_df
    

    def visualize_feature_importance(self, top_n=15):
        """Visualize feature importance rankings."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        top_features = self.importance_df.head(top_n)
        
        # Random Forest importance
        axes[0].barh(range(len(top_features)), top_features['rf_importance'])
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features['feature'], fontsize=8)
        axes[0].set_title('Random Forest Importance')
        axes[0].invert_yaxis()
        
        # Mutual Information
        axes[1].barh(range(len(top_features)), top_features['mi_importance'])
        axes[1].set_yticks(range(len(top_features)))
        axes[1].set_yticklabels(top_features['feature'], fontsize=8)
        axes[1].set_title('Mutual Information')
        axes[1].invert_yaxis()
        
        # Target Correlation
        axes[2].barh(range(len(top_features)), top_features['target_correlation'])
        axes[2].set_yticks(range(len(top_features)))
        axes[2].set_yticklabels(top_features['feature'], fontsize=8)
        axes[2].set_title('Target Correlation')
        axes[2].invert_yaxis()
        
        plt.tight_layout()
        plt.show()


    def recommend_features(self, importance_df, high_corr_pairs, top_n=20):
        """Recommend final feature set after removing multicollinearity."""
        print(f"\n🎯 Recommending final feature set...")
        
        # Start with top features
        recommended_features = []
        features_to_remove = set()
        
        # Identify features to remove due to high correlation
        for pair in high_corr_pairs:
            feat1, feat2 = pair['feature1'], pair['feature2']
            
            # Keep the more important feature
            feat1_rank = importance_df[importance_df['feature'] == feat1]['avg_rank'].iloc[0]
            feat2_rank = importance_df[importance_df['feature'] == feat2]['avg_rank'].iloc[0]
            
            if feat1_rank < feat2_rank:  # feat1 is more important
                features_to_remove.add(feat2)
                print(f"   Removing {feat2} (corr with {feat1}: {pair['correlation']:.3f})")
            else:
                features_to_remove.add(feat1)
                print(f"   Removing {feat1} (corr with {feat2}: {pair['correlation']:.3f})")
        
        # Select top features excluding removed ones
        for _, row in importance_df.iterrows():
            if row['feature'] not in features_to_remove:
                recommended_features.append(row['feature'])
            if len(recommended_features) >= top_n:
                break
        
        print(f"\n✅ Recommended features ({len(recommended_features)}):")
        for i, feat in enumerate(recommended_features[:15], 1):
            importance_rank = importance_df[importance_df['feature'] == feat]['avg_rank'].iloc[0]
            print(f"   {i:2d}. {feat} (rank: {importance_rank:.1f})")
        
        if len(recommended_features) > 15:
            print(f"   ... and {len(recommended_features)-15} more")
        
        return recommended_features
    

    def save_enhanced_dataset(self, merged_data, recommended_features, 
                         output_path="./data/artifacts/ml_ready",
                         file_prefix="enhanced"):
    
        # Prepare final dataset with recommended features + target
        final_features = recommended_features + ['cd']  # Add target variable
        enhanced_dataset = merged_data[['run'] + final_features].copy()
        
        # Save main dataset
        output_path = Path(output_path)
        main_file = output_path / f"{file_prefix}_dataset.csv"
        enhanced_dataset.to_csv(main_file, index=False)