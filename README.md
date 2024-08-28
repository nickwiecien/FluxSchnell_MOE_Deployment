# FLUX Schnell - Managed Online Endpoint Deployment (Azure ML)

Repository contains sample code demonstrating how to deploy [Black Forest Lab's FLUX schnell model](https://huggingface.co/black-forest-labs/FLUX.1-schnell) to a GPU-backed [Managed Online Endpoint](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online?view=azureml-api-2) in Azure ML.

### Prerequisites
- Access to an Azure Machine Learning workspace with the ability to deploy managed online endpoints
- Minimum of 24 cores of quota for Standard NCADSA100v4 Family Cluster Dedicated vCPUs (Note: alternative VM skus can be used for deployment, this quota is only required to run the sample notebooks as is.)

### Getting Started
- Update the missing environment variables and execute all cells in [01_Deployment.ipynb](./01_Deployment.ipynb). This will create your managed online deployment.
- Update the missing environment variables and execute all cells in [02_Testing.ipynb](./02_Testing.ipynb). This will submit 10 sample prompts to your newly created endpoint to verify operation.
