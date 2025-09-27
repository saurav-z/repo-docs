# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Accelerate your machine learning workflows with pre-built, optimized Docker images from AWS, designed for training and serving models across various frameworks.** (Link back to original repo: [https://github.com/aws/deep-learning-containers](https://github.com/aws/deep-learning-containers))

Key Features:

*   **Framework Support:** Includes TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Optimized Environments:** Pre-configured with necessary libraries like Nvidia CUDA (for GPU) and Intel MKL (for CPU).
*   **Deployment Options:** Compatible with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Container Registry:** Images available in Amazon Elastic Container Registry (Amazon ECR).

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the Framework Version](#upgrading-the-framework-version)
*   [Adding Artifacts to Your Build Context](#adding-artifacts-to-your-build-context)
*   [Adding a Package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

### Getting Started

Follow these steps to build and test the DLCs on platforms like Amazon SageMaker, EC2, ECS, and EKS.

**Example: Building an MXNet GPU Python3 Training Container**

1.  **Prerequisites:**
    *   An active AWS account.
    *   Configure the AWS CLI for access to your account. Recommended: Use an IAM role.  Necessary IAM permissions are listed in the original README.
    *   Create an ECR repository (e.g., "beta-mxnet-training") in your desired region (e.g., us-west-2).
    *   Install and configure the [Docker](https://docs.docker.com/get-docker/) client.

2.  **Setup:**
    *   Clone the repository.
    *   Set the following environment variables:
        ```shell script
        export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
        export REGION=us-west-2
        export REPOSITORY_NAME=beta-mxnet-training
        ```
    *   Log in to ECR:
        ```shell script
        aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
        ```
    *   Create a virtual environment and install dependencies:
        ```shell script
        python3 -m venv dlc
        source dlc/bin/activate
        pip install -r src/requirements.txt
        ```
    *   Perform the initial setup:
        ```shell script
        bash src/setup.sh mxnet
        ```

### Building Your Image

The build process uses `buildspec.yml` files and follows a specific Dockerfile path structure.

1.  **Build All Images:** Build all Dockerfiles defined in the buildspec.yml using:
    ```shell script
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  **Build a Single Image:** Build a specific image by specifying parameters:
    ```shell script
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
3.  **Parameters:**  Use `--image_types`, `--device_types`, and `--py_versions` to specify the images you want to build. These are comma-separated lists.  Valid values are described in the original README.

### Upgrading the Framework Version

To upgrade a framework version (e.g., MXNet 1.6.0 to 1.7.0):

1.  **Modify `buildspec.yml`:** Update the `version` field in the framework's `buildspec.yml` file.
2.  **Ensure Dockerfile:**  Verify the existence of the corresponding Dockerfile (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  **Build Container:**  Build the container using the steps outlined in the "Building your Image" section.

### Adding Artifacts to Your Build Context

Add files to the build context for use within your Docker images:

1.  **Specify Context:** Add the file details under the `context` key in the `buildspec.yml` file. You can define it in:
    *   The general `context` section (available to all images).
    *   `training_context` or `inference_context` (for specific image types).
    *   Within a specific image's context section.
2.  **Build Container:**  Rebuild your container.

### Adding a Package

To add a package to your image:

1.  **Modify Dockerfile:**  Add the `RUN ${PIP} install` command with the package name in your Dockerfile (e.g., within the MXNet Dockerfile).
2.  **Build Container:** Rebuild your container.

### Running Tests Locally

Run tests locally to validate changes before deployment:

1.  **Setup:**
    *   Ensure you have the desired DLC images available (e.g., pulled from ECR).
    *   Install test requirements:  `pip install -r src/requirements.txt && pip install -r test/requirements.txt`
    *   Set the following environment variables:
        ```shell script
        export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
        export PYTHONPATH=$(pwd)/src
        export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
        ```
2.  **Test Directory:** Change your directory to `test/dlc_tests`.
3.  **Run All Tests (EC2/ECS):**
    ```shell script
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto
    ```
    Remove `-n=auto` to run sequentially.
4.  **Run All Tests (EKS):**
    ```shell script
    cd ../
    export TEST_TYPE=eks
    python test/testrunner.py
    ```
5.  **Run Specific Test File:**
    ```shell script
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```
6.  **Run Specific Test Function:**
    ```shell script
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```
7.  **Run SageMaker Local Mode Tests:** Launch an EC2 instance, clone the repo, and run the specified pytest commands.
8.  **Run SageMaker Remote Tests:** Create an IAM role, set the correct permissions, and run the relevant pytest commands within the appropriate directory.
9.  **Run SageMaker Benchmark Tests:**
    *   Create `sm_benchmark_env_settings.config` and set necessary environment variables.
    *   Run  `source sm_benchmark_env_settings.config`.
    *   Run tests using `python test/testrunner.py` or specific pytest commands.