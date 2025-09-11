# AWS Deep Learning Containers: Optimize Your Machine Learning Workloads

Get started quickly with pre-built, optimized Docker images for training and serving your machine learning models.  [View the original repository on GitHub](https://github.com/aws/deep-learning-containers).

**Key Features:**

*   **Pre-built and Optimized:** Leverage pre-configured environments for TensorFlow, TensorFlow 2, PyTorch, and MXNet, eliminating the need for manual setup.
*   **GPU and CPU Support:** Benefit from optimized libraries like Nvidia CUDA (for GPU instances) and Intel MKL (for CPU instances) for enhanced performance.
*   **Amazon ECR Availability:** Easily access and deploy containers from the Amazon Elastic Container Registry (Amazon ECR).
*   **SageMaker Integration:** Seamlessly integrate with Amazon SageMaker for training, inference, and model transformation tasks.
*   **Cross-Platform Compatibility:** Works well on Amazon EC2, Amazon ECS, and Amazon EKS.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the Framework Version](#upgrading-the-framework-version)
*   [Adding Artifacts to Your Build Context](#adding-artifacts-to-your-build-context)
*   [Adding a Package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

## Getting Started

This section provides instructions to build and test the DLCs on various platforms like Amazon SageMaker, EC2, ECS, and EKS. The example focuses on building an MXNet GPU Python3 training container.

**Prerequisites:**

*   An AWS account with appropriate permissions (IAM roles are recommended). Suggested IAM policies are provided in the original README.
*   An ECR repository (e.g., "beta-mxnet-training" in us-west-2)
*   Docker client setup on your system.

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd deep-learning-containers
    ```
2.  **Set environment variables:**
    ```bash
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
3.  **Login to ECR:**
    ```bash
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
4.  **Create a virtual environment and install requirements:**
    ```bash
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
5.  **Perform initial setup:**
    ```bash
    bash src/setup.sh mxnet
    ```

## Building Your Image

The Dockerfiles are located based on the framework, training/inference type, version, python version and processor (e.g., mxnet/training/docker/1.6.0/py3/Dockerfile.gpu). This section provides details on how to build your custom images.

**Steps:**

1.  **Build all images from the buildspec.yml:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  **Build a single image:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
3.  **Arguments for image customization:** `--image_types`, `--device_types`, and `--py_versions`.

## Upgrading the Framework Version

How to upgrade the framework version in the buildspec.yml file.

1.  **Modify `buildspec.yml`:** Update the `version` key (e.g., change `1.6.0` to `1.7.0`).
2.  **Ensure Dockerfile exists:** Verify the Dockerfile for the new version exists (e.g., mxnet/docker/1.7.0/py3/Dockerfile.gpu).
3.  **Build the container:** Follow the build image instructions.

## Adding Artifacts to Your Build Context

Copy artifacts from your build context to the image.

1.  **Add to `buildspec.yml`:** Define the artifact in the `context` section, specifying the `source` file and `target` location within the container.
2.  **Context levels:**  Use `context`, `training_context`, or `inference_context` depending on the required scope.
3.  **Build the container:**  Follow the build image instructions.

## Adding a Package

This outlines how to add a package to your image.

1.  **Modify the Dockerfile:** Add the package installation command (e.g., `pip install octopush`) to the Dockerfile.
2.  **Build the container:**  Follow the build image instructions.

## Running Tests Locally

This section provides instructions to run tests on various platforms.

1.  **Install test requirements:**
    ```bash
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Set environment variables:** Define `DLC_IMAGES`, `CODEBUILD_RESOLVED_SOURCE_VERSION`, and `PYTHONPATH`.
    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Navigate to the test directory:**
    ```bash
    cd test/dlc_tests
    ```
4.  **Run tests:** Use `pytest` commands for specific platforms and test files:
    *   **EC2:** `pytest -s -rA ec2/ -n=auto`
    *   **ECS:** `pytest -s -rA ecs/ -n=auto`
    *   **EKS:**  `cd ../; export TEST_TYPE=eks; python test/testrunner.py`
    *   **Specific test file:** `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py`
    *   **Specific test function:** `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu`
5.  **SageMaker Local Mode Tests**
    *   Clone the github branch
    *   Login into the ECR repo where the new docker images built exist
    *   Run the test using `pytest` commands in the appropriate directory.
6.  **SageMaker Remote Tests**
    *   Create an IAM role with name “SageMakerRole” in the above account and add the AWS Manged policies
    *   Run the test using `pytest` commands in the appropriate directory.
7.  **SageMaker Benchmark Tests**
    *   Create `sm_benchmark_env_settings.config` file and add the below:
    *   Run the test using `pytest` commands in the appropriate directory.