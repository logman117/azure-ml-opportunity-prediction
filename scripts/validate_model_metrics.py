"""
Model Metrics Validation Script

Validates that trained model meets minimum performance requirements before
deployment to production. This prevents deploying underperforming models.

Validation Criteria:
- Accuracy >= 85%
- AUC >= 0.65
- Precision >= 0.80 (for positive class)
- Recall >= 0.70 (for positive class)

Usage:
    python scripts/validate_model_metrics.py --job-name <azure-ml-job-name>
    python scripts/validate_model_metrics.py --metrics-file outputs/production/production_model_metadata.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

# Validation thresholds - adjust these for your use case
VALIDATION_THRESHOLDS = {
    "accuracy": 0.85,      # Minimum 85% accuracy
    "auc": 0.65,           # Minimum 0.65 AUC
    "precision": 0.80,     # Minimum 80% precision
    "recall": 0.70,        # Minimum 70% recall
    "f1": 0.70            # Minimum 0.70 F1 score
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('validate_model_metrics')


def get_metrics_from_azure_ml_job(job_name: str) -> Dict[str, float]:
    """
    Retrieve metrics from Azure ML job.

    Args:
        job_name: Name of the Azure ML training job

    Returns:
        Dictionary of metric names to values

    Raises:
        RuntimeError: If job not found or metrics unavailable
    """
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        logger.info(f"Connecting to Azure ML to retrieve job: {job_name}")

        # Get Azure ML client
        subscription_id = os.environ['AZURE_SUBSCRIPTION_ID']
        resource_group = os.environ['AZURE_RESOURCE_GROUP']
        workspace_name = os.environ['AZURE_WORKSPACE_NAME']

        credential = DefaultAzureCredential()
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

        # Get job details
        job = ml_client.jobs.get(job_name)

        if job.status != "Completed":
            raise RuntimeError(f"Job status is '{job.status}', not 'Completed'")

        logger.info(f"Job status: {job.status}")

        # Try to get metrics from MLflow
        try:
            import mlflow

            # Get run ID from job
            run_id = job.name

            # Set MLflow tracking URI to Azure ML
            mlflow.set_tracking_uri("azureml://")

            # Get metrics from MLflow
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)

            metrics = {}
            for key, value in run.data.metrics.items():
                # Extract final metrics (prefixed with 'final_')
                if key.startswith('final_'):
                    metric_name = key.replace('final_', '')
                    metrics[metric_name] = value

            if not metrics:
                raise RuntimeError("No metrics found with 'final_' prefix")

            logger.info(f"Retrieved {len(metrics)} metrics from MLflow")
            return metrics

        except Exception as e:
            logger.warning(f"Failed to get metrics from MLflow: {e}")
            raise RuntimeError("Could not retrieve metrics from Azure ML job")

    except Exception as e:
        logger.error(f"Error retrieving metrics from Azure ML: {e}")
        raise


def get_metrics_from_file(file_path: str) -> Dict[str, float]:
    """
    Load metrics from local JSON file.

    Args:
        file_path: Path to JSON file containing metrics

    Returns:
        Dictionary of metric names to values

    Example file structure:
        {
            "final_metrics": {
                "accuracy": 0.8698,
                "auc": 0.7381,
                "precision": 0.8123,
                "recall": 0.7845,
                "f1": 0.7982
            }
        }
    """
    logger.info(f"Loading metrics from file: {file_path}")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Try to find metrics in common locations in the JSON
    if 'final_metrics' in data:
        metrics = data['final_metrics']
    elif 'metrics' in data:
        metrics = data['metrics']
    else:
        # Assume the entire file is metrics
        metrics = data

    logger.info(f"Loaded {len(metrics)} metrics from file")
    return metrics


def validate_metrics(metrics: Dict[str, float], thresholds: Dict[str, float]) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate metrics against thresholds.

    Args:
        metrics: Dictionary of actual metric values
        thresholds: Dictionary of minimum required values

    Returns:
        Tuple of (is_valid, validation_results)

    Example:
        >>> validate_metrics({"accuracy": 0.87, "auc": 0.74}, {"accuracy": 0.85, "auc": 0.65})
        (True, {"accuracy": {"value": 0.87, "threshold": 0.85, "pass": True}, ...})
    """
    logger.info("Validating metrics against thresholds...")
    logger.info("=" * 80)

    validation_results = {}
    all_passed = True

    for metric_name, threshold in thresholds.items():
        if metric_name not in metrics:
            logger.warning(f"Metric '{metric_name}' not found in results")
            validation_results[metric_name] = {
                "value": None,
                "threshold": threshold,
                "pass": False,
                "status": "MISSING"
            }
            all_passed = False
            continue

        value = metrics[metric_name]
        passed = value >= threshold

        validation_results[metric_name] = {
            "value": value,
            "threshold": threshold,
            "pass": passed,
            "status": "PASS" if passed else "FAIL"
        }

        # Log result with color indicators
        status_icon = "✓" if passed else "✗"
        status_text = "PASS" if passed else "FAIL"
        logger.info(f"{status_icon} {metric_name.upper():12} {status_text:6} | Value: {value:.4f} | Threshold: {threshold:.4f}")

        if not passed:
            all_passed = False

    logger.info("=" * 80)

    return all_passed, validation_results


def generate_validation_report(validation_results: Dict[str, Any], output_dir: Path = None) -> str:
    """
    Generate a detailed validation report.

    Args:
        validation_results: Results from validate_metrics()
        output_dir: Directory to save report (optional)

    Returns:
        Path to saved report file
    """
    report = {
        "validation_timestamp": None,
        "overall_status": "PASS" if all(r["pass"] for r in validation_results.values()) else "FAIL",
        "metrics_validated": len(validation_results),
        "metrics_passed": sum(1 for r in validation_results.values() if r["pass"]),
        "metrics_failed": sum(1 for r in validation_results.values() if not r["pass"]),
        "results": validation_results
    }

    # Add timestamp
    from datetime import datetime
    report["validation_timestamp"] = datetime.now().isoformat()

    # Save report if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_file = output_dir / "model_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved to: {report_file}")
        return str(report_file)

    return ""


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(
        description="Validate model metrics against thresholds"
    )

    # Input source: either Azure ML job or local file
    parser.add_argument(
        '--job-name',
        type=str,
        help='Azure ML job name to retrieve metrics from'
    )
    parser.add_argument(
        '--metrics-file',
        type=str,
        help='Path to local JSON file containing metrics'
    )

    # Optional: Custom thresholds
    parser.add_argument(
        '--accuracy-threshold',
        type=float,
        default=VALIDATION_THRESHOLDS['accuracy'],
        help=f'Minimum accuracy (default: {VALIDATION_THRESHOLDS["accuracy"]})'
    )
    parser.add_argument(
        '--auc-threshold',
        type=float,
        default=VALIDATION_THRESHOLDS['auc'],
        help=f'Minimum AUC (default: {VALIDATION_THRESHOLDS["auc"]})'
    )
    parser.add_argument(
        '--precision-threshold',
        type=float,
        default=VALIDATION_THRESHOLDS['precision'],
        help=f'Minimum precision (default: {VALIDATION_THRESHOLDS["precision"]})'
    )
    parser.add_argument(
        '--recall-threshold',
        type=float,
        default=VALIDATION_THRESHOLDS['recall'],
        help=f'Minimum recall (default: {VALIDATION_THRESHOLDS["recall"]})'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/validation',
        help='Directory to save validation report'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.job_name and not args.metrics_file:
        logger.error("Must provide either --job-name or --metrics-file")
        parser.print_help()
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("MODEL METRICS VALIDATION")
    logger.info("=" * 80)

    try:
        # Get metrics from specified source
        if args.job_name:
            logger.info(f"Source: Azure ML Job '{args.job_name}'")
            metrics = get_metrics_from_azure_ml_job(args.job_name)
        else:
            logger.info(f"Source: Local File '{args.metrics_file}'")
            metrics = get_metrics_from_file(args.metrics_file)

        logger.info(f"Metrics retrieved: {list(metrics.keys())}")

        # Update thresholds with command-line arguments
        thresholds = {
            "accuracy": args.accuracy_threshold,
            "auc": args.auc_threshold,
            "precision": args.precision_threshold,
            "recall": args.recall_threshold,
        }

        # Validate metrics
        is_valid, validation_results = validate_metrics(metrics, thresholds)

        # Generate report
        report_path = generate_validation_report(validation_results, Path(args.output_dir))

        # Print summary
        logger.info("=" * 80)
        if is_valid:
            logger.info("✓ VALIDATION PASSED: Model meets all performance requirements")
            logger.info("=" * 80)
            logger.info("Model is ready for production deployment!")
            sys.exit(0)
        else:
            logger.error("✗ VALIDATION FAILED: Model does not meet performance requirements")
            logger.info("=" * 80)
            logger.error("Model should NOT be deployed to production.")
            logger.error("Review validation report for details.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
