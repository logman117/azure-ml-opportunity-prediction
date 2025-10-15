"""
Submit production training job to Azure ML
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
import json

def get_ml_client():
    """Get Azure ML client"""
    subscription_id = os.environ['AZURE_SUBSCRIPTION_ID']
    resource_group = os.environ['AZURE_RESOURCE_GROUP']
    workspace_name = os.environ['AZURE_WORKSPACE_NAME']

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    return ml_client

def submit_training_job():
    """Submit production training job to Azure ML"""
    print("="*80)
    print("SUBMITTING PRODUCTION TRAINING JOB TO AZURE ML")
    print("="*80)

    ml_client = get_ml_client()

    # Job configuration
    compute_name = os.environ.get('COMPUTE_CLUSTER_NAME', 'gpu-cluster')
    environment_name = os.environ.get('ENVIRONMENT_NAME', 'opportunity-prediction-env')
    build_number = os.environ.get('BUILD_BUILDNUMBER', 'local')

    job_name = f"opportunity-prediction-prod-{datetime.now().strftime('%Y%m%d-%H%M%S')}-build-{build_number}"

    print(f"Job Name: {job_name}")
    print(f"Compute: {compute_name}")
    print(f"Environment: {environment_name}")

    # Create job
    job = command(
        code=".",
        command="python train_production.py",
        environment=f"{environment_name}@latest",
        compute=compute_name,
        display_name=job_name,
        experiment_name="opportunity-prediction-production",
        environment_variables={
            "DATA_READ_FUNCTION_URL": os.environ.get('DATA_READ_FUNCTION_URL', ''),
            "DATA_READ_KEY": os.environ.get('DATA_READ_KEY', ''),
            "MLFLOW_TRACKING_URI": "azureml://",
        },
        tags={
            "model_type": "production",
            "architecture": "MLP",
            "build_number": build_number
        }
    )

    # Submit job
    print("\nSubmitting job to Azure ML...")
    submitted_job = ml_client.jobs.create_or_update(job)

    print(f"\nJob submitted successfully!")
    print(f"Job Name: {submitted_job.name}")
    print(f"Job Status: {submitted_job.status}")
    print(f"Studio URL: {submitted_job.studio_url}")

    # Save job info
    job_info = {
        "job_name": submitted_job.name,
        "job_id": submitted_job.id,
        "status": submitted_job.status,
        "studio_url": submitted_job.studio_url,
        "submitted_at": datetime.now().isoformat()
    }

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    job_file = output_dir / f"production_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(job_file, 'w') as f:
        json.dump(job_info, f, indent=2)

    print(f"\nJob info saved to: {job_file}")

    return submitted_job

if __name__ == "__main__":
    try:
        job = submit_training_job()
        print("\n" + "="*80)
        print("JOB SUBMISSION COMPLETE")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: Job submission failed: {e}")
        sys.exit(1)
