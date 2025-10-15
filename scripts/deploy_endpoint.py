"""
Deploy model to Azure ML managed online endpoint
"""

import os
import sys
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential

def get_ml_client():
    """Get Azure ML client"""
    subscription_id = os.environ['AZURE_SUBSCRIPTION_ID']
    resource_group = os.environ['AZURE_RESOURCE_GROUP']
    workspace_name = os.environ['AZURE_WORKSPACE_NAME']

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    return ml_client

def create_scoring_script():
    """Create scoring script for deployment"""
    scoring_script = '''
import json
import joblib
import torch
import numpy as np
import logging
import os

def init():
    """Initialize model and scaler"""
    global model, scaler, device

    import mlflow.pytorch

    # Load model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    model = mlflow.pytorch.load_model(model_path)

    # Load scaler
    scaler_path = os.path.join(model_path, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logging.info("Model initialized successfully")

def run(raw_data):
    """Score incoming requests"""
    try:
        data = json.loads(raw_data)

        # Expected features
        feature_names = [
            'timestamp_days', 'is_duplicate', 'project_size',
            'Excluded', 'Premium_Tier_1', 'Premium_Tier_2',
            'Premium_Tier_3', 'Premium_Tier_4', 'Premium_Score',
            'Segment_Type_A', 'Segment_Type_B',
            'Segment_Type_C', 'Segment_Unknown',
            'Segment_Type_D', 'Segment_Type_E'
        ]

        # Extract features
        features = np.array([[data.get(f, 0.0) for f in feature_names]], dtype=np.float32)

        # Scale features
        features_scaled = scaler.transform(features)

        # Convert to tensor
        input_tensor = torch.FloatTensor(features_scaled).unsqueeze(1).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()

        # Return prediction
        result = {
            "probability": float(probability),
            "prediction": 1 if probability > 0.5 else 0,
            "confidence": float(abs(probability - 0.5) * 2)
        }

        return json.dumps(result)

    except Exception as e:
        logging.error(f"Scoring error: {str(e)}")
        return json.dumps({"error": str(e)})
'''

    output_dir = Path("outputs/deployment")
    output_dir.mkdir(parents=True, exist_ok=True)

    score_file = output_dir / "score.py"
    with open(score_file, 'w') as f:
        f.write(scoring_script)

    print(f"Scoring script created: {score_file}")
    return score_file

def deploy_model(model_name: str, model_version: str, endpoint_name: str = "opportunity-prediction-endpoint"):
    """Deploy model to managed online endpoint"""
    print("="*80)
    print("DEPLOYING MODEL TO AZURE ML ENDPOINT")
    print("="*80)

    ml_client = get_ml_client()

    print(f"Model: {model_name}:{model_version}")
    print(f"Endpoint: {endpoint_name}")

    # Create scoring script
    score_script = create_scoring_script()

    # Create or update endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Opportunity prediction model endpoint",
        auth_mode="key",
        tags={"model": model_name}
    )

    print("\nCreating/updating endpoint...")
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint '{endpoint_name}' is ready")

    # Create deployment
    deployment_name = "production"

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=f"{model_name}:{model_version}",
        code_configuration=CodeConfiguration(
            code=str(score_script.parent),
            scoring_script=score_script.name
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1
    )

    print(f"\nCreating deployment '{deployment_name}'...")
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Set traffic to 100%
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"\nDeployment complete!")
    print(f"Endpoint: {endpoint_name}")
    print(f"Deployment: {deployment_name}")

    # Get scoring URI and key
    endpoint_info = ml_client.online_endpoints.get(endpoint_name)
    print(f"\nScoring URI: {endpoint_info.scoring_uri}")

    keys = ml_client.online_endpoints.get_keys(endpoint_name)
    print(f"Primary Key: {keys.primary_key[:20]}...")

    return endpoint_info

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy model to Azure ML endpoint")
    parser.add_argument("--model-name", default="opportunity-prediction-model", help="Model name")
    parser.add_argument("--model-version", default="1", help="Model version")
    parser.add_argument("--endpoint-name", default="opportunity-prediction-endpoint", help="Endpoint name")

    args = parser.parse_args()

    try:
        endpoint = deploy_model(args.model_name, args.model_version, args.endpoint_name)
        print("\n" + "="*80)
        print("DEPLOYMENT COMPLETE")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
