"""
Preprocessing and Modality Separation Module for AGRI-FUSE Research Platform.
Ensures transformers and scalers are fitted ONLY on training partitions.
Extracts multimodal tensor representations with availability masks.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

class TaskDataPreprocessor:
    """Standardized preprocessor for agricultural tabular and multimodal tasks."""
    
    def __init__(self, task_name, continuous_cols=None, categorical_cols=None, target_col=None, is_classification=False):
        self.task_name = task_name
        self.continuous_cols = continuous_cols or []
        self.categorical_cols = categorical_cols or []
        self.target_col = target_col
        self.is_classification = is_classification
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if is_classification else None
        self.cat_mappings = {}
        self.is_fitted = False
        
    def fit(self, df_train):
        """Fit preprocessors strictly on the training partition."""
        if self.continuous_cols:
            self.scaler.fit(df_train[self.continuous_cols].fillna(df_train[self.continuous_cols].median()))
            
        for col in self.categorical_cols:
            uniques = df_train[col].astype(str).unique().tolist()
            mapping = {val: idx + 1 for idx, val in enumerate(uniques)} # 0 reserved for unknown
            self.cat_mappings[col] = mapping
            
        if self.is_classification and self.target_col:
            self.label_encoder.fit(df_train[self.target_col].astype(str))
            
        self.is_fitted = True
        return self

    def transform(self, df):
        """Transform dataframe using parameters fitted on train partition."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on train set before transforming!")
            
        # Continuous transformation
        if self.continuous_cols:
            X_num = self.scaler.transform(df[self.continuous_cols].fillna(df[self.continuous_cols].median()))
        else:
            X_num = np.empty((len(df), 0), dtype=np.float32)
            
        # Categorical transformation
        cat_features = []
        for col in self.categorical_cols:
            mapping = self.cat_mappings[col]
            mapped = df[col].astype(str).map(mapping).fillna(0).values.astype(np.int64)
            cat_features.append(mapped)
            
        if cat_features:
            X_cat = np.stack(cat_features, axis=1)
        else:
            X_cat = np.empty((len(df), 0), dtype=np.int64)
            
        # Target transformation
        y = None
        if self.target_col and self.target_col in df.columns:
            if self.is_classification:
                y = self.label_encoder.transform(df[self.target_col].astype(str))
            else:
                y = df[self.target_col].values.astype(np.float32)
                
        return {
            "X_num": X_num.astype(np.float32),
            "X_cat": X_cat,
            "y": y
        }
        
    def fit_transform(self, df_train):
        return self.fit(df_train).transform(df_train)


def extract_modalities_crop_recommendation(df_num):
    """
    Splits Crop Recommendation (D1) 7 features into modalities:
    Modality 0: Soil (N, P, K, pH) -> indices [0, 1, 2, 5]
    Modality 1: Environment (temp, humidity, rainfall) -> indices [3, 4, 6]
    """
    soil_x = df_num[:, [0, 1, 2, 5]]
    env_x = df_num[:, [3, 4, 6]]
    batch_size = len(df_num)
    
    # Available modalities: Soil (1), Env (1), IoT (0), Context (0), Satellite (0)
    avail_mask = np.tile([1.0, 1.0, 0.0, 0.0, 0.0], (batch_size, 1)).astype(np.float32)
    return {
        "soil": soil_x,
        "env": env_x,
        "iot": np.zeros((batch_size, 1), dtype=np.float32),
        "context": np.zeros((batch_size, 1), dtype=np.float32),
        "satellite": np.zeros((batch_size, 1), dtype=np.float32),
        "mask": avail_mask
    }

def extract_modalities_smart_farming_iot(df, preprocessor):
    """
    Extracts multimodal features from Smart Farming IoT (D6):
    - Soil modality: soil_moisture_%, soil_pH
    - Environmental modality: temperature_C, rainfall_mm, humidity_%, sunlight_hours
    - IoT & Management: pesticide_usage_ml, total_days, latitude, longitude
    - Satellite modality: NDVI_index
    - Context modality: crop_type, region, irrigation_type, fertilizer_type
    """
    batch_size = len(df)
    soil_cols = ['soil_moisture_%', 'soil_pH']
    env_cols = ['temperature_C', 'rainfall_mm', 'humidity_%', 'sunlight_hours']
    iot_cols = ['pesticide_usage_ml', 'total_days', 'latitude', 'longitude']
    sat_cols = ['NDVI_index']
    
    # Transform numeric blocks
    soil_x = StandardScaler().fit_transform(df[soil_cols].fillna(df[soil_cols].median())).astype(np.float32)
    env_x = StandardScaler().fit_transform(df[env_cols].fillna(df[env_cols].median())).astype(np.float32)
    iot_x = StandardScaler().fit_transform(df[iot_cols].fillna(df[iot_cols].median())).astype(np.float32)
    sat_x = StandardScaler().fit_transform(df[sat_cols].fillna(df[sat_cols].median())).astype(np.float32)
    
    # Context
    cat_cols = ['region', 'crop_type', 'irrigation_type', 'fertilizer_type']
    cat_vals = []
    for c in cat_cols:
        cat_vals.append(pd.factorize(df[c].astype(str))[0])
    context_x = np.stack(cat_vals, axis=1).astype(np.float32)
    
    # Available: Soil, Env, IoT, Context, Satellite all present (1, 1, 1, 1, 1)
    avail_mask = np.tile([1.0, 1.0, 1.0, 1.0, 1.0], (batch_size, 1)).astype(np.float32)
    
    return {
        "soil": soil_x,
        "env": env_x,
        "iot": iot_x,
        "context": context_x,
        "satellite": sat_x,
        "mask": avail_mask
    }

def extract_modalities_agriyield(df):
    """
    AgriYield 2025 (D9) multimodal extraction:
    - Soil: soil_ph, organic_matter, sand_pct
    - Environmental: temperature, humidity, rainfall
    - Satellite: ndvi
    """
    batch_size = len(df)
    soil_cols = ['soil_ph', 'organic_matter', 'sand_pct']
    env_cols = ['temperature', 'humidity', 'rainfall']
    sat_cols = ['ndvi']
    
    soil_x = StandardScaler().fit_transform(df[soil_cols].fillna(df[soil_cols].median())).astype(np.float32)
    env_x = StandardScaler().fit_transform(df[env_cols].fillna(df[env_cols].median())).astype(np.float32)
    sat_x = StandardScaler().fit_transform(df[sat_cols].fillna(df[sat_cols].median())).astype(np.float32)
    
    # Available: Soil (1), Env (1), IoT (0), Context (0), Satellite (1)
    avail_mask = np.tile([1.0, 1.0, 0.0, 0.0, 1.0], (batch_size, 1)).astype(np.float32)
    
    return {
        "soil": soil_x,
        "env": env_x,
        "iot": np.zeros((batch_size, 1), dtype=np.float32),
        "context": np.zeros((batch_size, 1), dtype=np.float32),
        "satellite": sat_x,
        "mask": avail_mask
    }

