# AWS Deep Learning Containers: Build, Train, and Deploy Your Machine Learning Models

AWS Deep Learning Containers (DLCs) provide pre-built Docker images for training and serving machine learning models using popular frameworks like TensorFlow, PyTorch, and MXNet.  [Learn more on GitHub](https://github.com/aws/deep-learning-containers).

**Key Features:**

*   **Optimized Environments:** Pre-configured with TensorFlow, MXNet, CUDA (for GPU instances), and Intel MKL (for CPU instances) for optimal performance.
*   **Framework Support:**  Offers images for TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Integration with AWS Services:** Designed for use with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Pre-built Images:** Available in Amazon Elastic Container Registry (Amazon ECR).

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the Framework Version](#upgrading-the-framework-version)
*   [Adding Artifacts to Your Build Context](#adding-artifacts-to-your-build-context)
*   [Adding a Package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

## Getting Started

This section outlines the setup to build and test the DLCs on Amazon SageMaker, EC2, ECS, and EKS.

**Prerequisites:**

*   An active AWS account.  Configure your environment using the AWS CLI.  We recommend using an IAM role. Ensure your IAM role has the necessary permissions:
    *   `AmazonEC2ContainerRegistryFullAccess`
    *   `AmazonEC2FullAccess`
    *   `AmazonEKSClusterPolicy`
    *   `AmazonEKSServicePolicy`
    *   `AmazonEKSServiceRolePolicy`
    *   `AWSServiceRoleForAmazonEKSNodegroup`
    *   `AmazonSageMakerFullAccess`
    *   `AmazonS3FullAccess`

*   An ECR repository.  Create a repository (e.g., "beta-mxnet-training" in `us-west-2`).
*   Docker installed and configured on your system.

**Setup Instructions (Example: MXNet GPU Python3 Training):**

1.  Clone the repository and set environment variables:

    ```bash
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```

2.  Log in to ECR:

    ```bash
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```

3.  Create and activate a virtual environment and install requirements:

    ```bash
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```

4.  Perform initial setup:

    ```bash
    bash src/setup.sh mxnet
    ```

## Building Your Image

Dockerfiles follow a specific pattern (e.g., `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`).  The `buildspec.yml` file specifies these paths. Modify `buildspec.yml` to build specific framework versions.

1.  To build all Dockerfiles defined in `buildspec.yml` locally:

    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

2.  To build a single image:

    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```

3.  Possible values for `--image_types`, `--device_types`, and `--py_versions` are:
    *   `--image_types`: `training/inference`
    *   `--device_types`: `cpu/gpu`
    *   `--py_versions`: `py2/py3`

4.  Example: Build all GPU training containers:

    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

## Upgrading the Framework Version

1.  Modify the `buildspec.yml` file. For example, to upgrade MXNet from version 1.6.0 to 1.7.0:
    ```yaml
    # mxnet/training/buildspec.yml
    # ...
    version: &VERSION 1.7.0  # Change this line
    # ...
    ```

2.  Ensure the corresponding Dockerfile exists (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).

3.  Build the container as described above.

## Adding Artifacts to Your Build Context

1.  Add the artifact to the `buildspec.yml` file, under the `context` key:

    ```yaml
    # mxnet/training/buildspec.yml
    context:
      README.xyz:  # Object name (can be anything)
        source: README-context.rst  # Path to the file
        target: README.rst  # Name in the build context
    ```

2.  Use `training_context` or `inference_context` to apply the artifact to those specific image types.

3.  Apply it to a specific image using its own context key.

4.  Build the container as described above.

## Adding a Package

1.  Modify the Dockerfile to install the desired package (e.g., `octopush`):

    ```dockerfile
    # mxnet/training/docker/1.6.0/py3/Dockerfile.gpu
    # ...
    RUN ${PIP} install --no-cache --upgrade \
        awscli \
        octopush   # Add the package here
    ```

2.  Build the container as described above.

## Running Tests Locally

Run tests locally to avoid using external resources and speed up iteration.

1.  Ensure you have the images you want to test (pull them from ECR).  Change directory to the cloned repo. Install test requirements:

    ```bash
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```

2.  Export environment variables:

    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

3.  Change directory to the tests:

    ```bash
    cd test/dlc_tests
    ```

4.  To run all tests for a platform (EC2, ECS):

    ```bash
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto
    ```

5.  To run a specific test file:

    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

6.  To run a specific test function:

    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```
7. Run SageMaker local mode, remote mode and benchmark tests. Instructions are provided in the original README.