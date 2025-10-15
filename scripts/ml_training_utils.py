"""
Essential ML Training Utilities
Core functions for training opportunity prediction models
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import logging
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
import os
import requests
import joblib

warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
class Config:
    """Basic configuration for ML training"""
    # ML Features list - 15 features total
    ML_FEATURES = [
        'timestamp_days',  # Days since reference date
        'is_duplicate',
        'project_size',
        # Premium/Priority features (6 features)
        'Excluded',
        'Premium_Tier_1',
        'Premium_Tier_2',
        'Premium_Tier_3',
        'Premium_Tier_4',
        'Premium_Score',
        # Segment features (6 features - one-hot encoded)
        'Segment_Type_A',
        'Segment_Type_B',
        'Segment_Type_C',
        'Segment_Unknown',
        'Segment_Type_D',
        'Segment_Type_E'
    ]

    RANDOM_SEED = 42
    CV_FOLDS = 5
    EARLY_STOPPING_PATIENCE = 10
    MAX_EPOCHS = 100

def set_all_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(output_dir: Optional[Path] = None) -> logging.Logger:
    """Set up logging for training"""
    logger = logging.getLogger('ml_training')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger

def fetch_from_table_storage(logger: logging.Logger) -> pd.DataFrame:
    """Fetch training data from cloud storage via API with pagination"""
    import requests
    import time

    url = f"{os.environ['DATA_READ_FUNCTION_URL']}/api/GetMlTrainingData"
    headers = {"x-functions-key": os.environ['DATA_READ_KEY']}

    all_data = []
    batch_size = 10000
    batch_num = 1
    continuation_token = None
    total_fetched = 0
    max_total_records = 180000

    try:
        logger.info(f"Starting paginated fetch from URL: {url}")
        logger.info(f"Batch size: {batch_size:,} records per request")

        while True:
            if total_fetched >= max_total_records:
                logger.info(f"Reached maximum record limit of {max_total_records:,}")
                break

            remaining_records = max_total_records - total_fetched
            current_batch_size = min(batch_size, remaining_records)

            params = {"limit": current_batch_size}
            if continuation_token:
                params["continuationToken"] = continuation_token

            logger.info(f"Fetching batch {batch_num} (size: {current_batch_size:,})")

            # Retry logic
            max_retries = 8
            retry_delay = 10

            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers=headers, params=params, timeout=600, stream=True)
                    response.raise_for_status()
                    data = response.json()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Batch {batch_num} attempt {attempt + 1} failed. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 300)
                    else:
                        logger.error(f"Batch {batch_num} failed after {max_retries} attempts")
                        raise

            if not isinstance(data, dict):
                logger.error(f"Unexpected response format for batch {batch_num}")
                break

            batch_data = data.get('data', [])
            batch_count = len(batch_data)
            has_more = data.get('hasMore', False)
            continuation_token = data.get('continuationToken')

            logger.info(f"Batch {batch_num}: Received {batch_count:,} records")

            if not batch_data:
                logger.info(f"No data in batch {batch_num}, stopping")
                break

            all_data.extend(batch_data)
            total_fetched += batch_count

            if not has_more:
                logger.info(f"Pagination complete. Last batch had {batch_count:,} records")
                break

            batch_num += 1

        logger.info(f"Total fetched: {total_fetched:,} records in {batch_num} batches")

        if not all_data:
            logger.warning("No data received")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch from storage: {e}")
        raise

def load_and_prepare_data(data_source: str, logger: logging.Logger) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Load pre-cleaned ML data from storage or CSV.
    Data is already cleaned during ingestion.
    """
    try:
        # Step 1: Load data
        if data_source == 'table_storage':
            logger.info("Loading data from cloud storage")
            df = fetch_from_table_storage(logger)
        else:
            logger.info(f"Loading data from CSV: {data_source}")
            df = pd.read_csv(data_source)

        if df.empty:
            logger.error("No data loaded")
            return None, None

        initial_rows = len(df)
        logger.info(f"Loaded {initial_rows} records")

        # Step 2: Use pre-calculated action_taken column as target
        if 'action_taken' not in df.columns:
            logger.error("action_taken column not found in data!")
            return None, None

        y = df['action_taken'].astype(int)

        # Verify we have both classes
        target_distribution = y.value_counts().to_dict()
        logger.info(f"Target distribution: {target_distribution}")

        if len(y.unique()) == 1:
            logger.error(f"Only one class in target! All values are {y.unique()[0]}")

        # Step 3: Verify ML features exist
        missing_features = []
        for feature in Config.ML_FEATURES:
            if feature not in df.columns:
                missing_features.append(feature)
                df[feature] = 0

        if missing_features:
            logger.warning(f"Added missing features: {missing_features}")

        # Step 4: Select features
        X = df[Config.ML_FEATURES].copy()
        X = X.astype('float32').fillna(0.0)

        logger.info(f"Final dataset: X={X.shape}, y={y.shape}")
        logger.info(f"Features: {X.columns.tolist()}")

        return X, y

    except Exception as e:
        logger.error(f"Error in load_and_prepare_data: {str(e)}")
        return None, None

# ===== MODEL DEFINITIONS =====
class FlexibleRNN(nn.Module):
    """Flexible RNN/LSTM/GRU model for binary classification"""

    def __init__(self, input_size, hidden_size, num_layers, dropout, model_type='LSTM'):
        super(FlexibleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type

        # Choose RNN type
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:  # RNN
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = rnn_out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class FlexibleMLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular data
    Wide & shallow architecture
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(FlexibleMLP, self).__init__()

        layers = []
        current_size = input_size

        # Build hidden layers with constant width
        for i in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = hidden_size

        # Output layer
        layers.append(nn.Linear(current_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Squeeze sequence dimension if present
        if len(x.shape) == 3:
            x = x.squeeze(1)
        return self.network(x)

def create_model(model_type: str, input_size: int, hidden_size: int,
                num_layers: int, dropout: float) -> nn.Module:
    """Create model based on type"""
    if model_type in ['RNN', 'LSTM', 'GRU']:
        return FlexibleRNN(input_size, hidden_size, num_layers, dropout, model_type)
    elif model_type == 'MLP':
        return FlexibleMLP(input_size, hidden_size, num_layers, dropout)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def create_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float,
                    weight_decay: float = 1e-5) -> torch.optim.Optimizer:
    """Create optimizer"""
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics"""
    metrics = {}

    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
    except Exception:
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'avg_precision']:
            metrics[metric] = 0.0

    return metrics

def save_percentile_mapping(probabilities, output_dir, logger):
    """Create and save probability distribution for percentile calculation"""
    try:
        probs_array = np.array(probabilities, dtype=np.float32)
        sorted_probs = np.sort(probs_array)

        quantiles = np.arange(0, 1.0001, 0.0001)
        quantile_values = np.percentile(sorted_probs, quantiles * 100)

        percentile_mapping = {
            "quantiles": quantiles.tolist(),
            "values": quantile_values.tolist(),
            "min_prob": float(np.min(sorted_probs)),
            "max_prob": float(np.max(sorted_probs)),
            "median_prob": float(np.median(sorted_probs)),
            "mean_prob": float(np.mean(sorted_probs)),
            "std_prob": float(np.std(sorted_probs)),
            "num_samples": len(sorted_probs),
            "created_at": datetime.now().isoformat()
        }

        percentile_path = output_dir / "percentile_mapping.json"
        with open(percentile_path, 'w') as f:
            json.dump(percentile_mapping, f, indent=2)

        logger.info(f"Percentile mapping saved: {percentile_path}")
        return percentile_path

    except Exception as e:
        logger.error(f"Failed to save percentile mapping: {str(e)}")
        return None

def run_single_configuration(
    config: Dict[str, Any],
    data_path: Optional[str] = None,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    device: str = 'cuda',
    log_to_mlflow: bool = True
) -> Dict[str, Any]:
    """
    Core training function for a single hyperparameter configuration
    """

    logger = setup_logging()

    try:
        set_all_seeds(Config.RANDOM_SEED)

        # Use provided data OR load from path
        if X is not None and y is not None:
            logger.info("Using pre-cleaned data for training")
        else:
            if data_path is None:
                raise ValueError("Must provide either (X, y) or data_path")
            X, y = load_and_prepare_data(data_path, logger)
            if X is None or y is None:
                raise ValueError("Failed to load training data")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_scaled = np.array(X_scaled, dtype=np.float32)
        y_array = np.array(y, dtype=np.float32)

        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Training parameters
        model_type = config.get('model_type', 'MLP')
        hidden_size = config.get('hidden_size', 384)
        num_layers = config.get('num_layers', 1)
        dropout = config.get('dropout', 0.1)
        learning_rate = config.get('learning_rate', 0.001)
        batch_size = config.get('batch_size', 64)
        optimizer_name = config.get('optimizer', 'Adam')
        weight_decay = config.get('weight_decay', 1e-5)
        epochs = config.get('epochs', 50)

        logger.info(f"Training {model_type} with config: {config}")

        # Create datasets (80/20 split)
        full_dataset = TensorDataset(
            torch.FloatTensor(X_scaled).unsqueeze(1),
            torch.FloatTensor(y_array)
        )

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        model = create_model(model_type, X_scaled.shape[1], hidden_size, num_layers, dropout)
        model = model.to(device)

        optimizer = create_optimizer(model, optimizer_name, learning_rate, weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        training_history = []

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0
            val_predictions = []
            val_probabilities = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > 0.5).astype(int)

                    val_predictions.extend(preds)
                    val_probabilities.extend(probs)
                    val_targets.extend(batch_y.cpu().numpy())

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            epoch_metrics = calculate_metrics(val_targets, val_predictions, val_probabilities)
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **epoch_metrics
            })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.get('early_stopping_patience', 10):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final evaluation
        model.eval()
        final_predictions = []
        final_probabilities = []
        final_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                final_predictions.extend(preds)
                final_probabilities.extend(probs)
                final_targets.extend(batch_y.cpu().numpy())

        final_metrics = calculate_metrics(final_targets, final_predictions, final_probabilities)

        # Collect training probabilities for percentile mapping
        logger.info("Collecting probabilities for percentile mapping...")
        train_probabilities = []

        with torch.no_grad():
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()

                if isinstance(probs, np.ndarray):
                    if probs.ndim == 0:
                        train_probabilities.append(float(probs))
                    else:
                        train_probabilities.extend(probs.tolist())
                else:
                    train_probabilities.append(float(probs))

        logger.info(f"Collected {len(train_probabilities)} training probabilities")

        # Save artifacts
        output_dir = Path("outputs")
        model_dir = output_dir / "model"
        output_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)

        percentile_path = save_percentile_mapping(train_probabilities, model_dir, logger)

        scaler_path = model_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'model_config': config,
                'feature_names': list(X.columns) if hasattr(X, 'columns') else Config.ML_FEATURES,
                'metrics': final_metrics
            }, f, indent=2)

        # Save with MLflow if available
        if log_to_mlflow:
            try:
                import mlflow
                import mlflow.pytorch

                for metric_name, metric_value in final_metrics.items():
                    mlflow.log_metric(f"final_{metric_name}", metric_value)

                extra_files = []
                if scaler_path.exists():
                    extra_files.append(str(scaler_path.absolute()))
                if percentile_path and percentile_path.exists():
                    extra_files.append(str(percentile_path.absolute()))

                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    registered_model_name=None,
                    extra_files=extra_files,
                    pip_requirements=[
                        "torch>=2.0.0",
                        "scikit-learn>=1.3.0",
                        "pandas>=2.0.0",
                        "numpy>=1.24.0",
                        "joblib>=1.3.0"
                    ]
                )
                logger.info("Model saved in MLflow format")

            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        logger.info(f"Training completed - AUC: {final_metrics['auc']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")

        return {
            'config': config,
            'metrics': final_metrics,
            'training_history': training_history,
            'model_path': str(model_dir),
            'scaler_path': str(scaler_path),
            'config_path': str(config_path),
            'percentile_path': str(percentile_path) if percentile_path else None,
            'device': str(device),
            'success': True,
            'training_completed': True
        }

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return {
            'config': config,
            'metrics': {'auc': 0.0, 'accuracy': 0.0},
            'error': str(e),
            'success': False,
            'training_completed': False
        }

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
