import torch 
import numpy as np 

class Normalizer:
    def __init__(self, method='standard'):
        self.scalers = []
        self.method = method
        self.global_stats = {}
        # self.fit(data)
    
    def fit(self, X):
        n_samples, n_timesteps, n_features = X.shape  # (n_samples, 1200, 352)
        X_np = X.numpy()

        # Compute global statistics across all samples, timesteps, and features
        X_flat = X_np.reshape(-1, n_features)  # [n_samples * n_timesteps, 352]
        self.global_stats = {
            'global_min': np.min(X_flat),
            'global_max': np.max(X_flat),
            'global_std': np.std(X_flat, axis=0).tolist(),      # convert to list for JSON
            'global_median': np.median(X_flat, axis=0).tolist(),
            'global_99th': np.percentile(X_flat, 99, axis=0).tolist(),
            'per_voxel_min': np.min(X_flat, axis=0).tolist(),
            'per_voxel_max': np.max(X_flat, axis=0).tolist()
        }
        quartiles = np.nanpercentile(X_flat, [25, 75], axis=0)
        self.global_stats['global_iqr'] = (quartiles[1, :] - quartiles[0, :]).tolist()

        # # ✅ Save stats to JSON
        # with open("global_stats_ukbiobank.json", "w") as f:
        #     json.dump(self.global_stats, f, indent=2)
        # print("Saved!!!")

        # Initialize scalers for 'standard' and 'minmax' methods
        if self.method in ['standard', 'minmax']:
            scaler_class = StandardScaler if self.method == 'standard' else MinMaxScaler
            for f in range(n_features):
                scaler = scaler_class()
                X_f = X_np[:, :, f].reshape(-1, 1)
                scaler.fit(X_f)
                self.scalers.append(scaler)
    
    def transform(self, X):
        n_samples, n_timesteps, n_features = X.shape
        X_np = X.numpy()
        X_transformed = np.zeros_like(X_np)
        
        if self.method == 'raw':
            X_transformed = X_np  # No normalization
        elif self.method == 'robust_scaler':
            # Subtract mean per parcel, then scale by median and IQR
            X_transformed = X_np - X_np.mean(axis=1, keepdims=True)
            self.global_stats['global_iqr'] = [x if x != 0 else 30000 for x in self.global_stats['global_iqr']]
            X_transformed = (X_transformed - self.global_stats['global_median']) / self.global_stats['global_iqr']
        elif self.method == 'all_patient_all_voxel':
            # Min-max across all data
            X_transformed = (X_np - self.global_stats['global_min']) / (self.global_stats['global_max'] - self.global_stats['global_min'])
        elif self.method == 'per_patient_all_voxel':
            # Min-max per sample (subject)
            for i in range(n_samples):
                X_transformed[i] = (X_np[i] - X_np[i].min()) / (X_np[i].max() - X_np[i].min())
        elif self.method == 'per_patient_per_voxel':
            # Min-max per parcel per sample
            for i in range(n_samples):
                for f in range(n_features):
                    v_min, v_max = X_np[i, :, f].min(), X_np[i, :, f].max()
                    if v_max > v_min:
                        X_transformed[i, :, f] = (X_np[i, :, f] - v_min) / (v_max - v_min)
        elif self.method == 'per_voxel_all_patient':
            # Min-max per parcel across all samples
            for f in range(n_features):
                v_min, v_max = self.global_stats['per_voxel_min'][f], self.global_stats['per_voxel_max'][f]
                if v_max > v_min:
                    X_transformed[:, :, f] = (X_np[:, :, f] - v_min) / (v_max - v_min)
        elif self.method == 'subtract_mean':
            # Subtract mean per parcel
            X_transformed = X_np - X_np.mean(axis=1, keepdims=True)
        elif self.method == 'subtract_mean_global_std':
            # Subtract mean per parcel, divide by global std
            X_transformed = (X_np - X_np.mean(axis=1, keepdims=True)) / self.global_stats['global_std']
        elif self.method == 'subtract_mean_99th_percentile':
            # Subtract mean per parcel, divide by global 99th percentile
            X_transformed = (X_np - X_np.mean(axis=1, keepdims=True)) / self.global_stats['global_99th']
        elif self.method in ['standard', 'minmax']:
            # Existing scaler-based methods
            for f in range(n_features):
                X_f = X_np[:, :, f].reshape(-1, 1)
                X_transformed[:, :, f] = self.scalers[f].transform(X_f).reshape(n_samples, n_timesteps)
        
        return torch.tensor(X_transformed, dtype=torch.float32)