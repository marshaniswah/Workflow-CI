# Workflow-CI

This repository implements Criteria 3 for Dicoding SMSML using GitHub Actions and MLflow Project.

## Project

**Natural Scene Classification with MLflow Monitoring Pipeline**

Dataset:

```text
Intel Image Classification
```

Classes:

```text
buildings
forest
glacier
mountain
sea
street
```

## Required Structure

```text
Workflow-CI/
  .github/
    workflows/
      ci.yml
  MLProject/
    modelling.py
    conda.yaml
    MLProject
    dataset_preprocessing/
    DockerHub.txt
```

## How CI Works

The workflow runs when triggered manually or when files under `MLProject` change.

The workflow performs:

1. checkout repository
2. setup Conda environment
3. run model training using `mlflow run`
4. upload MLflow runs, model, and training artifacts as GitHub Actions artifacts
5. build Docker image using `mlflow models build-docker --env-manager local`
6. push Docker image to Docker Hub

## Required GitHub Secrets

Add these secrets in GitHub:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

Use a Docker Hub access token, not your plain password.

## Manual Local Test

```powershell
cd MLProject
mlflow run . -e main -P dataset_dir=dataset_preprocessing -P epochs=2 -P batch_size=16 -P image_size=160
```

## Docker Hub

After a successful Advanced workflow run, update `MLProject/DockerHub.txt` with the final Docker Hub image URL.

Expected image format:

```text
https://hub.docker.com/r/<dockerhub_username>/natural-scene-classifier
```

## Checklist

- Repository visibility is Public.
- `MLProject` folder exists.
- `.github/workflows/ci.yml` exists.
- `mlflow run` succeeds in GitHub Actions.
- GitHub Actions uploads `mlflow-training-output`.
- Docker image is built using `mlflow models build-docker --env-manager local`.
- Docker image is pushed to Docker Hub.
