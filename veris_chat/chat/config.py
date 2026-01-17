"""
Configuration loader for veris_chat.

Loads config.yaml and .env, returning a dict with model names, Qdrant settings,
and AWS credentials. Supports AWS SSO fallback when env vars are empty.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


def load_config(
    config_path: str = "config.yaml",
    env_path: str = ".env",
) -> Dict[str, Any]:
    """
    Load configuration from config.yaml and .env files.

    Args:
        config_path: Path to config.yaml file.
        env_path: Path to .env file.

    Returns:
        Dict containing merged configuration with keys:
        - models: embedding_model, generation_model
        - qdrant: collection_name, vector_size, url, api_key
        - aws: region, access_key_id, secret_access_key, session_token, use_sso
        - logging: level, log_prefix, console_output
        - chunking: chunk_size, strategy, overlap
        - paths: csv_data, pdfs, parsed, chunks, embeddings
    """
    # Load .env file
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file)

    # Load config.yaml
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Build AWS config with SSO fallback
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_session_token = os.getenv("AWS_SESSION_TOKEN", "")
    
    # Region from .env (default us-east-1 for Opus 4.5 access with us.* prefix)
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    # Determine if using SSO (when env vars are empty)
    # On EC2 with Instance Profile, credentials come from metadata service automatically
    use_sso = not (aws_access_key and aws_secret_key)

    config = {
        "models": yaml_config.get("models", {}),
        "qdrant": {
            **yaml_config.get("qdrant", {}),
            "url": os.getenv("QDRANT_URL", ""),
            "api_key": os.getenv("QDRANT_API_KEY", ""),
        },
        "aws": {
            "region": aws_region,
            "access_key_id": aws_access_key,
            "secret_access_key": aws_secret_key,
            "session_token": aws_session_token,
            "use_sso": use_sso,
        },
        "logging": yaml_config.get("logging", {}),
        "chunking": yaml_config.get("chunking", {}),
        "paths": yaml_config.get("paths", {}),
    }

    return config


def get_bedrock_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build kwargs dict for LlamaIndex Bedrock components.

    If AWS credentials are empty, returns minimal kwargs to use SSO/default credentials.
    This works for both local SSO development and EC2 Instance Profile deployment.

    Args:
        config: Configuration dict from load_config().

    Returns:
        Dict of kwargs for BedrockEmbedding or Bedrock LLM initialization.
    """
    aws_cfg = config.get("aws", {})
    # Default to us-east-1 for Opus 4.5 access (RMIT SCP requires us.* prefix)
    kwargs = {"region_name": aws_cfg.get("region", "us-east-1")}

    # Only add explicit credentials if not using SSO/Instance Profile
    if not aws_cfg.get("use_sso", True):
        kwargs["aws_access_key_id"] = aws_cfg.get("access_key_id")
        kwargs["aws_secret_access_key"] = aws_cfg.get("secret_access_key")
        if aws_cfg.get("session_token"):
            kwargs["aws_session_token"] = aws_cfg.get("session_token")

    return kwargs
