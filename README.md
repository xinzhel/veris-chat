# VERIS RAG

Multi-PDF RAG system using LlamaIndex with AWS Bedrock and Qdrant.

## Setup
```bash
conda env create -f environment.yaml
conda activate veris_vectordb
aws sso login  # if using SSO credentials
```

## Test Connectivity
```bash
python script/test_connectivity.py
```
