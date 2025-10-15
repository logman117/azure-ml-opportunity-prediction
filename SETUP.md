# Project Setup Guide

## Summary of Created Files

This directory contains a complete, generalized version of your ML opportunity prediction system ready for public GitHub.

### What Was Changed from Original

**Company-Specific → Generic:**
- Project-specific terminology → Generic "opportunity prediction" or "record"
- `ProjectCheckpoints` → `IncrementalCheckpoints`
- `PROJECT_API_URL` → `EXTERNAL_API_URL`
- `PROJECT_API_KEY` → `EXTERNAL_API_KEY`
- `project-search-service` → `your-search-service`
- `project-index` → `your-index-name`
- GraphDB references → External API
- All company/customer-specific names generalized

**What Still Works:**
- ✅ All Azure Functions code
- ✅ Complete ML training pipeline
- ✅ Model deployment scripts
- ✅ Data ingestion and processing
- ✅ MLP neural network architecture
- ✅ MLflow integration
- ✅ Azure ML job submission

## File Structure

```
github files/
├── README.md                          # Main project documentation
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore patterns
├── .env.example                       # Environment variable template
├── CODE_OF_CONDUCT.md                 # Community guidelines
├── CONTRIBUTING.md                    # Contribution guidelines
├── SECURITY.md                        # Security policy
├── ARCHITECTURE.md                    # Technical architecture
├── SETUP.md                           # This file
│
├── requirements.txt                   # Azure Functions dependencies
├── requirements_production.txt        # ML training dependencies
├── host.json                          # Function app configuration
├── .funcignore                        # Function deployment exclusions
│
├── data_ingestion/                    # Real-time data ingestion function
│   ├── __init__.py                    # Main function code
│   └── function.json                  # Timer trigger config (5 min)
│
├── shared/                            # Shared modules
│   ├── __init__.py
│   ├── data_search.py                 # Cloud search operations
│   ├── api_client.py                  # External API client
│   └── common.py                      # Utilities & data prep
│
├── scripts/                           # ML training & deployment
│   ├── ml_training_utils.py           # Core ML functions & models
│   ├── submit_production_job.py       # Azure ML job submission
│   ├── register_model.py              # Model registration
│   └── deploy_endpoint.py             # Endpoint deployment
│
├── configs/                           # Configuration files
│   └── production_config.json         # Production model config
│
└── train_production.py                # Main training script

```

## Quick Start

### 1. Clone to Your Repository

```bash
# Copy all files from "github files/" to your new repo
cp -r "github files/"* /path/to/your/new/repo/
cd /path/to/your/new/repo/
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
# Edit .env with your Azure credentials
```

### 3. Install Dependencies

```bash
# For Azure Functions
pip install -r requirements.txt

# For ML Training
pip install -r requirements_production.txt
```

### 4. Test Locally

```bash
# Start Azure Functions locally
func start

# Run ML training locally
python train_production.py --data-path ./data/sample_data.csv
```

### 5. Deploy to Azure

```bash
# Deploy functions
func azure functionapp publish <your-function-app-name>

# Submit ML training job
python scripts/submit_production_job.py
```

## Key Features Preserved

### Data Pipeline
- ✅ Real-time ingestion every 5 minutes
- ✅ Checkpoint-based incremental processing
- ✅ Failed ID retry mechanism
- ✅ Batch processing (100 records per batch)
- ✅ Base64 ID decoding with error handling

### ML System
- ✅ MLP architecture (384 hidden units, 1 layer)
- ✅ 15 engineered features
- ✅ 87% accuracy, 0.74 AUC
- ✅ Cross-validation training
- ✅ MLflow experiment tracking
- ✅ Automated model registration
- ✅ Managed endpoint deployment

### Automation
- ✅ Azure DevOps pipeline ready
- ✅ Automatic retraining capability
- ✅ CI/CD integration points
- ✅ Model versioning

## Environment Variables Needed

### Data Ingestion
```bash
SEARCH_SERVICE_NAME=your-search-service
SEARCH_INDEX_NAME=your-index-name
SEARCH_API_KEY=your-api-key
AZURE_STORAGE_CONNECTION_STRING=your-connection-string
EXTERNAL_API_URL=your-api-endpoint
EXTERNAL_API_KEY=your-api-key
```

### ML Training
```bash
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_WORKSPACE_NAME=your-workspace-name
COMPUTE_CLUSTER_NAME=your-compute-cluster
ENVIRONMENT_NAME=your-ml-environment
DATA_READ_FUNCTION_URL=your-data-function-url
DATA_READ_KEY=your-data-function-key
```

## What to Update for Your Use Case

1. **README.md**: Update with your specific business domain
2. **Model Features**: Modify `ML_FEATURES` in `scripts/ml_training_utils.py` for your data
3. **Data Schema**: Update `extract_*` functions in `shared/common.py` for your fields
4. **Partition Logic**: Modify `build_partition_filter()` for your data categories
5. **Hyperparameters**: Adjust `production_config.json` after your own tuning

## Files You Can Safely Customize

- `README.md` - Add your project details
- `configs/production_config.json` - Tune hyperparameters
- `shared/common.py` - Adjust feature extraction
- `.env.example` - Add your specific env vars
- `ARCHITECTURE.md` - Document your specific setup

## Files to Keep As-Is

- Core ML logic in `scripts/ml_training_utils.py`
- Model architectures (FlexibleMLP, FlexibleRNN)
- Azure Functions structure
- Deployment scripts

## Next Steps

1. ✅ Review all files to ensure no sensitive data
2. ✅ Update README.md with your project specifics
3. ✅ Test locally with sample data
4. ✅ Create GitHub repository
5. ✅ Push code to GitHub
6. ✅ Add CI/CD workflows if desired

## License

This code is under MIT License - you're free to use, modify, and distribute.

## Support

For questions about the code:
- Check ARCHITECTURE.md for technical details
- See CONTRIBUTING.md for development guidelines
- Review documentation in README.md

---

**Created**: January 2025
**Purpose**: Production-ready ML opportunity prediction system
**Status**: Fully functional and generalized
