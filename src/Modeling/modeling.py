import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV


class Modelling:

    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_cols = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.Scaler = None

    def load_ml_data(self, artifact_path: str):
        '''Load the merged dataset'''
        df_path = Path(artifact_path) / "ml_ready" /"merged_dataset.csv"
        self.df = pd.read_csv(df_path)

        #Prepare features and target
        self.feature_cols = [col for col in self.df.columns if col not in ['run', 'Run', 'cd']]
        self.X = self.df[self.feature_cols]
        self.y = self.df['cd']

        #Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        #Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"Dataset loaded: {len(self.df)} samples, {len(self.feature_cols)} features")
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test, self.feature_cols
    
    
    def plot_linear_regression_results(self, y_true, y_pred):
        """Plot model results"""
        
        # Actual vs Predicted
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Cd')
        plt.ylabel('Predicted Cd')
        plt.title(f'Actual vs Predicted\nR² = {r2_score(y_true, y_pred):.3f}')

    
    def train_linear_regression(self,cv):
        """Train Linear Regression model"""

        print("\n" + "="*50)
        print("Linear Regression Model")
        print("="*50)
        

        #Train Model
        model = LinearRegression()
        model.fit(self.X_train_scaled, self.y_train)

        #Predictions
        y_train_preds = model.predict(self.X_train_scaled)
        y_test_preds = model.predict(self.X_test_scaled)

        #Evaluations
        train_r2 = r2_score(self.y_train, y_train_preds)
        test_r2 = r2_score(self.y_test, y_test_preds)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_preds))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_preds))

        print(f"\nModel Performance:")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")


        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='r2')
        print(f"\n3-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        #Feature importance(coefficients)
        feature_importance = pd.DataFrame({
            'Feature': self.feature_cols,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)

        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10)[['Feature', 'Coefficient']].to_string(index=False))

        # Visualization
        self.plot_linear_regression_results(self.y_test, y_test_preds)
        
        return model, train_r2, test_r2, feature_importance
    
    def plot_rf_results(self, y_true, y_pred, feature_importance):
        """Plot Random Forest results"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Actual vs Predicted
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[0].set_xlabel('Actual Cd')
        axes[0].set_ylabel('Predicted Cd')
        axes[0].set_title(f'RF: Actual vs Predicted\nR² = {r2_score(y_true, y_pred):.3f}')
        axes[0].grid(True, alpha=0.3)
        
        # Feature importance
        top_features = feature_importance.head(10)
        axes[1].barh(range(len(top_features)), top_features['Importance'])
        axes[1].set_yticks(range(len(top_features)))
        axes[1].set_yticklabels(top_features['Feature'])
        axes[1].set_xlabel('Feature Importance')
        axes[1].set_title('Top 10 Feature Importance')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    
    def train_random_forest(self, cv):
        """Train RF Model"""
        print("RANDOM FOREST MODEL")
        print("="*50)

        # Train model
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Evaluate
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        print(f"\nPerformance:")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='r2')
        print(f"3-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Plot results
        self.plot_rf_results(self.y_test, y_test_pred, feature_importance)
        
        return model, train_r2, test_r2, feature_importance
    

    def plot_xgb_results(self, y_true, y_pred, feature_importance):
        """Plot XGBoost results"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Actual vs Predicted
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[0].set_xlabel('Actual Cd')
        axes[0].set_ylabel('Predicted Cd')
        axes[0].set_title(f'XGB: Actual vs Predicted\nR² = {r2_score(y_true, y_pred):.3f}')
        axes[0].grid(True, alpha=0.3)
        
        # Feature importance
        top_features = feature_importance.head(10)
        axes[1].barh(range(len(top_features)), top_features['Importance'])
        axes[1].set_yticks(range(len(top_features)))
        axes[1].set_yticklabels(top_features['Feature'])
        axes[1].set_xlabel('Feature Importance')
        axes[1].set_title('Top 10 Feature Importance')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


    def train_xgboost(self, cv):
        """Train XGBoost model with regularization"""
        
        print("XGBOOST MODEL")
        print("="*50)
        
        
        # Train model with regularization
        model = xgb.XGBRegressor(
            n_estimators=80,
            max_depth=4,           # Shallow trees
            learning_rate=0.1,     # Conservative learning
            subsample=0.8,         # Row sampling
            colsample_bytree=0.8,  # Feature sampling
            reg_alpha=0.1,         # L1 regularization
            reg_lambda=1.0,        # L2 regularization
            random_state=42
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Evaluate
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        print(f"\nPerformance:")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='r2')
        print(f"5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Plot results
        self.plot_xgb_results(self.y_test, y_test_pred, feature_importance)
        
        return model, train_r2, test_r2, feature_importance
    
    def plot_ridge_lasso_results(self, y_true, ridge_pred, lasso_pred, ridge_r2, lasso_r2):
        """Plot Ridge vs Lasso results"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Ridge
        axes[0].scatter(y_true, ridge_pred, alpha=0.6)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[0].set_xlabel('Actual Cd')
        axes[0].set_ylabel('Predicted Cd')
        axes[0].set_title(f'Ridge: R² = {ridge_r2:.3f}')
        axes[0].grid(True, alpha=0.3)
        
        # Lasso
        axes[1].scatter(y_true, lasso_pred, alpha=0.6)
        axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[1].set_xlabel('Actual Cd')
        axes[1].set_ylabel('Predicted Cd')
        axes[1].set_title(f'Lasso: R² = {lasso_r2:.3f}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    
    def train_ridge_lasso(self, cv):
        """Train Ridge and Lasso models with CV"""
        
        print("RIDGE & LASSO REGRESSION")
        print("="*50)
        
        # Ridge with CV
        ridge_cv = RidgeCV(alphas=np.logspace(-3, 2, 50), cv=cv)
        ridge_cv.fit(self.X_train_scaled, self.y_train)
        
        ridge_pred = ridge_cv.predict(self.X_test_scaled)
        ridge_r2 = r2_score(self.y_test, ridge_pred)
        ridge_rmse = np.sqrt(mean_squared_error(self.y_test, ridge_pred))
        
        # Lasso with CV
        lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=cv, max_iter=2000)
        lasso_cv.fit(self.X_train_scaled, self.y_train)
        
        lasso_pred = lasso_cv.predict(self.X_test_scaled)
        lasso_r2 = r2_score(self.y_test, lasso_pred)
        lasso_rmse = np.sqrt(mean_squared_error(self.y_test, lasso_pred))
        
        print(f"\nRidge Results:")
        print(f"Best alpha: {ridge_cv.alpha_:.4f}")
        print(f"Test R²: {ridge_r2:.4f}")
        print(f"Test RMSE: {ridge_rmse:.4f}")
        
        print(f"\nLasso Results:")
        print(f"Best alpha: {lasso_cv.alpha_:.4f}")
        print(f"Test R²: {lasso_r2:.4f}")
        print(f"Test RMSE: {lasso_rmse:.4f}")
        
        # Feature selection by Lasso
        lasso_coefs = pd.DataFrame({
            'Feature': self.feature_cols,
            'Coefficient': lasso_cv.coef_
        })
        selected_features = lasso_coefs[lasso_coefs['Coefficient'] != 0]
        print(f"\nLasso selected {len(selected_features)} features:")
        print(selected_features.sort_values('Coefficient', key=abs, ascending=False).to_string(index=False))
        
        # Plot comparison
        self.plot_ridge_lasso_results(self.y_test, ridge_pred, lasso_pred, ridge_r2, lasso_r2)
        
        return ridge_cv, lasso_cv, ridge_r2, lasso_r2
        
