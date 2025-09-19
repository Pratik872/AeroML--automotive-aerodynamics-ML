import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Import your modeling class
from src.Modeling.modeling import Modelling

def get_feature_slider(feature_name, feature_ranges, label=None, help_text=None):
    """Helper function to create slider with proper min/max ranges"""
    if label is None:
        label = feature_name.replace('_', ' ').title()
    
    feature_range = feature_ranges[feature_ranges['Feature'] == feature_name]
    
    if not feature_range.empty:
        min_val = float(feature_range['Min_Value'].iloc[0])
        max_val = float(feature_range['Max_Value'].iloc[0])
        
        # For binary features (0,1), use selectbox instead
        if min_val == 0.0 and max_val == 1.0:
            return st.selectbox(label, [0, 1], index=0, help=help_text)
        else:
            # For continuous features, use slider
            default_val = min_val + (max_val - min_val) / 2
            return st.slider(label, 
                           min_value=min_val, 
                           max_value=max_val, 
                           value=default_val,
                           help=help_text)
    else:
        st.error(f"Feature {feature_name} not found in ranges!")
        return 0.0

def load_feature_ranges():
    """Load min/max ranges for features from saved model artifacts"""
    try:
        model_artifacts = joblib.load('models/linear_regression_model.pkl')
        feature_max = model_artifacts['feature_max']
        feature_min = model_artifacts['feature_min'] 
        feature_cols = model_artifacts['feature_cols']
        
        # Convert pandas series to DataFrame
        ranges = pd.DataFrame({
            'Feature': feature_cols,
            'Min_Value': [feature_min[col] for col in feature_cols],
            'Max_Value': [feature_max[col] for col in feature_cols]
        })
        return ranges
        
    except FileNotFoundError:
        st.error("Model file not found!")
        return None
    except KeyError as e:
        st.error(f"Feature ranges not found in model file: {e}")
        return None

def create_drag_prediction_app():
    st.title("🚗 Automotive Drag Coefficient Prediction")
    st.markdown("Predict drag coefficients using traditional ML")
    
    # Load feature ranges
    feature_ranges = load_feature_ranges()
    
    # Create sidebar for model info
    st.sidebar.header("Model Information")
    
    try:
        model_artifacts = joblib.load('models/linear_regression_model.pkl')
        st.sidebar.success(f"✅ Model loaded successfully")
        st.sidebar.write(f"**Test R²:** {model_artifacts['test_r2']:.4f}")
        
    except FileNotFoundError:
        st.sidebar.error("❌ Model not found. Please train the model first.")
        return
    
    # Main input form
    st.header("Vehicle Configuration")
    
    # Create tabs for different parameter groups
    tab1, tab2, tab3 = st.tabs(["🚗 Vehicle Type", "🔧 Configuration", "📏 Dimensions"])
    
    with tab1:
        st.subheader("Vehicle Body Type")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fastback = st.selectbox("Fastback", [0, 1], index=0, 
                                  help="1 for fastback design, 0 otherwise")
        with col2:
            estate = st.selectbox("Estate", [0, 1], index=0,
                                help="1 for estate/wagon, 0 otherwise")
        with col3:
            notchback = st.selectbox("Notchback", [0, 1], index=1,
                                   help="1 for notchback sedan, 0 otherwise")
    
    with tab2:
        st.subheader("Aerodynamic Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            no_mirrors = st.selectbox("No Mirrors", [0, 1], index=0,
                                    help="1 for no mirrors, 0 with mirrors")
            smooth_underbody = st.selectbox("Smooth Underbody", [0, 1], index=0,
                                          help="1 for smooth, 0 for detailed underbody")
        
        with col2:
            closed_wheels = st.selectbox("Closed Wheels", [0, 1], index=0,
                                       help="1 for closed wheels, 0 for open")
            detailed_underbody = st.selectbox("Detailed Underbody", [0, 1], index=1,
                                            help="1 for detailed, 0 for simplified")
    
    with tab3:
        st.subheader("Vehicle Dimensions (mm)")
        
        # Get ranges for dimensional parameters
        dim_features = ['length_mm', 'width_mm', 'height_mm', 'wheelbase_mm', 
                       'front_track_mm', 'rear_track_mm', 'front_overhang_mm', 
                       'rear_overhang_mm', 'ride_height_mm']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            length_mm = get_feature_slider('Vehicle_Length', feature_ranges, "Length (mm)")
            width_mm = get_feature_slider('Vehicle_Width', feature_ranges, "Width (mm)")  
            height_mm = get_feature_slider('Vehicle_Height', feature_ranges, "Height (mm)")
        
        with col2:
            wheelbase_mm = get_feature_slider('Front_Planview', feature_ranges, "Front Planview")
            front_track_mm = get_feature_slider('Hood_Angle', feature_ranges, "Hood Angle")
            rear_track_mm = get_feature_slider('Approach_Angle', feature_ranges, "Approach Angle")
        
        with col3:
            front_overhang_mm = get_feature_slider('Front_Overhang', feature_ranges, "Front Overhang")
            rear_overhang_mm = get_feature_slider('Rear_Overhang', feature_ranges, "Rear Overhang")  
            ride_height_mm = get_feature_slider('Vehicle_Ride_Height', feature_ranges, "Ride Height")
    
    # Prediction button
    st.markdown("---")
    if st.button("🚀 Predict Drag Coefficient", type="primary"):
        # Prepare input data in correct order
        input_data = [
            fastback, estate, notchback, no_mirrors, smooth_underbody, 
            closed_wheels, detailed_underbody, length_mm, width_mm, height_mm, 
            wheelbase_mm, front_track_mm, rear_track_mm, front_overhang_mm, 
            rear_overhang_mm, ride_height_mm
        ]
        
        try:
            # Initialize modeling class and make prediction
            model_obj = Modelling()
            prediction = model_obj.predict_single_input(input_data)
            
            if prediction is not None:
                # Display results with comparison
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("**Predicted Cd**", f"{prediction:.6f}")
                
                with col2:
                    baseline = 0.879  # 87.9% as decimal approximation
                    if prediction < baseline * 0.3:  # Rough Cd range conversion
                        st.metric("**vs Baseline**", "Better", delta=f"-{abs(prediction - baseline*0.3):.4f}")
                    else:
                        st.metric("**vs Baseline**", "Higher", delta=f"+{abs(prediction - baseline*0.3):.4f}")
                
                with col3:
                    drag_category = "Low" if prediction < 0.28 else "Medium" if prediction < 0.32 else "High"
                    st.metric("**Drag Category**", drag_category)
                
                # Show input summary
                with st.expander("📊 Input Summary"):
                    input_df = pd.DataFrame({
                        'Feature': model_artifacts['feature_cols'],
                        'Value': input_data
                    })
                    st.dataframe(input_df, use_container_width=True)
                
                # Comparison with other approaches
                with st.expander("🔬 Project Results Comparison"):
                    results_df = pd.DataFrame({
                        'Approach': ['Traditional ML (Baseline)', 'Linear Regression', '3D CNN', 'PointNet', 'MeshCNN (Target)'],
                        'R² Score': ['87.9%', f'{model_artifacts["test_r2"]*100:.1f}%', 'Failed', 'Failed', 'Failed'],
                        'Status': ['✅ Completed', '✅ Completed', '❌ Overfitting', '❌ Overfitting', '❌ Overfitting']
                    })
                    st.dataframe(results_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("Make sure the model is properly trained and saved.")

def main():
    st.set_page_config(
        page_title="AeroML - Drag Prediction",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    create_drag_prediction_app()

if __name__ == "__main__":
    main()