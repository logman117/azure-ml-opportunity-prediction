"""
Register trained model in Azure ML Model Registry
"""

import os
import sys
import json
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes

def get_ml_client():
    """Get Azure ML client"""
    subscription_id = os.environ['AZURE_SUBSCRIPTION_ID']
    resource_group = os.environ['AZURE_RESOURCE_GROUP']
    workspace_name = os.environ['AZURE_WORKSPACE_NAME']

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    return ml_client

def register_model_from_job(job_name: str, model_name: str = "opportunity-prediction-model"):
    """Register model from completed training job"""
    print("="*80)
    print("REGISTERING MODEL IN AZURE ML")
    print("="*80)

    ml_client = get_ml_client()

    print(f"Job Name: {job_name}")
    print(f"Model Name: {model_name}")

    # Get job
    job = ml_client.jobs.get(job_name)
    print(f"Job Status: {job.status}")

    if job.status != "Completed":
        print(f"WARNING: Job status is {job.status}, not Completed")

    # Create model from job outputs
    model_path = f"azureml://jobs/{job_name}/outputs/artifacts/model"

    print(f"Model Path: {model_path}")

    # Register model
    model = Model(
        name=model_name,
        path=model_path,
        type=AssetTypes.MLFLOW_MODEL,
        description="Production opportunity prediction model - MLP architecture",
        tags={
            "framework": "pytorch",
            "architecture": "MLP",
            "task": "binary_classification",
            "training_job": job_name
        }
    )

    print("\nRegistering model...")
    registered_model = ml_client.models.create_or_update(model)

    print(f"\nModel registered successfully!")
    print(f"Model Name: {registered_model.name}")
    print(f"Model Version: {registered_model.version}")
    print(f"Model ID: {registered_model.id}")

    # Save registration info
    reg_info = {
        "model_name": registered_model.name,
        "model_version": registered_model.version,
        "model_id": registered_model.id,
        "job_name": job_name,
        "registered_at": str(registered_model.creation_context.created_at)
    }

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    reg_file = output_dir / "model_registration.json"
    with open(reg_file, 'w') as f:
        json.dump(reg_info, f, indent=2)

    print(f"\nRegistration info saved to: {reg_file}")

    return registered_model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register model from Azure ML job")
    parser.add_argument("--job-name", required=True, help="Name of completed training job")
    parser.add_argument("--model-name", default="opportunity-prediction-model", help="Name for registered model")

    args = parser.parse_args()

    try:
        model = register_model_from_job(args.job_name, args.model_name)
        print("\n" + "="*80)
        print("MODEL REGISTRATION COMPLETE")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: Model registration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
