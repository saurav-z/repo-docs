# AWS Deep Learning Containers: Build, Train, and Deploy Machine Learning Models

**Get up and running with optimized Docker images for deep learning, built by AWS for effortless model training and deployment.**  [Learn more at the original repository](https://github.com/aws/deep-learning-containers).

## Key Features:

*   **Pre-built and Optimized:** Ready-to-use Docker images for TensorFlow, TensorFlow 2, PyTorch, and MXNet, pre-configured with optimized libraries like CUDA and Intel MKL for peak performance.
*   **Flexible Deployment:** Designed for seamless integration with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Easy Access:** Available in the Amazon Elastic Container Registry (Amazon ECR) for simple deployment.
*   **Comprehensive Coverage:** Supports training, inference, and transform workloads.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the framework version](#upgrading-the-framework-version)
*   [Adding artifacts to your build context](#adding-artifacts-to-your-build-context)
*   [Adding a package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

## Getting Started

This section details the setup process for building and testing AWS Deep Learning Containers (DLCs) on various platforms, including Amazon SageMaker, EC2, ECS, and EKS. It provides an example using an MXNet GPU python3 training container.

**Prerequisites:**

*   An active AWS account with configured access (IAM user or role is recommended) with the following managed permissions:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
*   Docker client installed on your system.

**Steps:**

1.  **AWS Account Setup:** Ensure your AWS CLI is configured to access your AWS account.
2.  **Create ECR Repository:** Create an ECR repository with the name "beta-mxnet-training" in the us-west-2 region (or your desired region).
3.  **Clone and Configure:**
    ```shell
    git clone <repository-url>
    cd <repository-directory>
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
4.  **Docker Login:** Authenticate with ECR.
    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
5.  **Set Up Virtual Environment:** Create and activate a virtual environment and install the requirements.
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
6.  **Initial Setup:** Perform the initial setup for MXNet.
    ```shell
    bash src/setup.sh mxnet
    ```

## Building Your Image

The paths to the Dockerfiles follow a specific pattern and are defined in the buildspec.yml file within each framework's directory. Modify the buildspec.yml to specify the Dockerfile version.

1.  **Build All Images:** To build all images defined in the buildspec.yml, run:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  **Build Specific Images:** To build a single image:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet --image_types training --device_types cpu --py_versions py3
    ```
3.  **Arguments:** The `--image_types`, `--device_types`, and `--py_versions` arguments are comma-separated lists with the following possible values:
    *   `--image_types`: `training/inference`
    *   `--device_types`: `cpu/gpu`
    *   `--py_versions`: `py2/py3`
4.  **Example (GPU Training):**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet --image_types training --device_types gpu --py_versions py3
    ```

## Upgrading the framework version

1.  **Modify `buildspec.yml`:** Change the version number in the `buildspec.yml` file for the target framework (e.g., change `version: 1.6.0` to `version: 1.7.0`).
2.  **Dockerfile Location:** Ensure the Dockerfile for the new version exists in the correct directory (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  **Build Container:** Build the container using the instructions in the "Building Your Image" section.

## Adding artifacts to your build context

1.  **Add to `buildspec.yml`:** To copy artifacts into the build context, add the artifacts under the `context` key in the framework's `buildspec.yml` file.
2.  **Context Options:** You can add artifacts to all images, training images, inference images, or a specific image, depending on the context key used.
    *   `context`: for all images
    *   `training_context`:  for training images
    *   `inference_context`:  for inference images
3.  **Build Container:** Build the container using the instructions in the "Building Your Image" section.

## Adding a package

1.  **Modify Dockerfile:** Edit the Dockerfile (e.g., `mxnet/training/docker/1.6.0/py3/Dockerfile.gpu`) and add the desired package using the `RUN ${PIP} install` command.
2.  **Build Container:** Build the container using the instructions in the "Building Your Image" section.

## Running Tests Locally

Use pytest to run tests locally after pulling the needed images from ECR. This allows you to avoid using too many resources or waiting for a build to complete.

**Prerequisites:**

*   EC2 instance or local machine.
*   The `deep-learning-containers` repo cloned.
*   Docker images to be tested (pulled from ECR).
*   `pytest` and test requirements installed.

**Steps:**

1.  **Install Test Requirements:**
    ```shell
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Set Environment Variables:**
    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
    (Replace the ECR URIs with your image URIs.)
3.  **Navigate to Test Directory:**
    ```shell
    cd test/dlc_tests
    ```
4.  **Run All Tests (Series):**
    ```shell
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto
    #EKS
    cd ../
    export TEST_TYPE=eks
    python test/testrunner.py
    ```
    (Remove `-n=auto` to run sequentially.)
5.  **Run Specific Test File:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```
6.  **Run Specific Test Function:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```
7.  **SageMaker Local Mode Tests:** Follow the instructions provided to run SageMaker local mode tests.
8.  **SageMaker Remote Tests:** Run SageMaker remote tests using the provided commands after setting up the necessary IAM role and following the setup instructions.
9.  **SageMaker Benchmark Tests:** Run SageMaker benchmark tests using the provided commands after creating the `sm_benchmark_env_settings.config` and setting the necessary configurations.