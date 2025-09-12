# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Accelerate your machine learning workflows with pre-built, optimized Docker images for training and serving your models on AWS.**  [View the original repository](https://github.com/aws/deep-learning-containers)

AWS Deep Learning Containers (DLCs) offer a streamlined solution for deploying and managing machine learning models. These Docker images come pre-configured with popular deep learning frameworks, NVIDIA CUDA, and Intel MKL, providing an optimized environment for your workloads.

**Key Features:**

*   **Framework Support:**  Includes TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Optimized Environments:**  Pre-installed with NVIDIA CUDA (for GPU instances) and Intel MKL (for CPU instances) for performance.
*   **Amazon ECR Availability:** Easily accessible through the Amazon Elastic Container Registry (Amazon ECR).
*   **SageMaker Integration:**  Used as the default for Amazon SageMaker jobs (training, inference, etc.).
*   **Platform Compatibility:**  Tested and suitable for Amazon EC2, Amazon ECS, and Amazon EKS.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the Framework Version](#upgrading-the-framework-version)
*   [Adding Artifacts to Your Build Context](#adding-artifacts-to-your-build-context)
*   [Adding a Package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

## Getting Started

This section provides a quick guide on how to build and test the DLCs on platforms like Amazon SageMaker, EC2, ECS, and EKS.  As an example, the steps below demonstrate building an ***MXNet GPU python3 training*** container.

**Prerequisites:**

*   An active AWS account.  Set up your environment with the AWS CLI, ideally using an IAM role.  Recommended permissions include:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
*   An ECR repository.  Create one (e.g., "beta-mxnet-training" in us-west-2) using the AWS CLI.
*   Docker client installed and configured on your system.

**Steps:**

1.  Clone the repository and set environment variables:
    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
2.  Login to ECR:
    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
3.  Create and activate a virtual environment, and install dependencies.  Assuming your working directory is the cloned repo:
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
4.  Perform initial setup:
    ```shell
    bash src/setup.sh mxnet
    ```

## Building Your Image

The Dockerfiles are located according to a specific pattern: `/<framework>/training/docker/<version>/<python_version>/Dockerfile.<processor>`. Use the `buildspec.yml` file in the framework's training directory (e.g., `mxnet/training/buildspec.yml`) to control the build process.

1.  Build all Dockerfiles specified in the `buildspec.yml`:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  Build a single image:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
3.  The `--image_types`, `--device_types`, and `--py_versions` arguments accept comma-separated lists.

    *   `--image_types`: `training`, `inference`
    *   `--device_types`: `cpu`, `gpu`
    *   `--py_versions`: `py2`, `py3`

4.  Example: Build all GPU training containers:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

## Upgrading the Framework Version

1.  Modify the `buildspec.yml` file to reflect the new version.  Example: change `version: &VERSION 1.6.0` to `version: &VERSION 1.7.0`.
2.  Ensure the corresponding Dockerfile exists (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  Build the container.

## Adding Artifacts to Your Build Context

1.  To include artifacts (e.g., `README-context.rst`) in your build, add them to the `context` section of the `buildspec.yml` file.  Specify the `source` and `target` paths:
    ```yaml
    context:
      README.xyz:
        source: README-context.rst
        target: README.rst
    ```
2.  Use `training_context` or `inference_context` for images to include only training or inference images, respectively.
3.  To add artifacts to a single container image, include the `context` under the individual image configuration in `buildspec.yml`.
4.  Build the container.

## Adding a Package

1.  Modify the Dockerfile to install the package using `pip`.  Example: add `octopush` to the installation list:
    ```dockerfile
    RUN ${PIP} install --no-cache --upgrade \
        ...
        awscli \
        octopush
    ```
2.  Build the container.

## Running Tests Locally

Test your images locally using pytest.  This section describes how to set up and run the tests.

1.  Make sure you have the necessary images (pull them from ECR) and install test requirements:
    ```shell
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  Set environment variables:
    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  Change your working directory to the test directory:
    ```shell
    cd test/dlc_tests
    ```
4.  Run all tests for a given platform (EC2, ECS):
    ```shell
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto
    ```
    To run the tests sequentially remove `-n=auto`
    ```shell
    pytest -s -rA ec2/
    ```

    For EKS the command would be:
    ```shell
    cd ../
    export TEST_TYPE=eks
    python test/testrunner.py
    ```

5.  Run a specific test file:
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```
6.  Run a specific test function:
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

7. SageMaker local mode tests setup
   * Clone your github branch with changes
   * Login into the ECR repo where the new docker images built exist
   * Change to the appropriate directory based on framework and job type of the image being tested.
   * Run pytest tests:
      ```shell
      python3 -m pytest -v integration/local --region us-west-2 \
      --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
       --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
       --py-version 3
      ```

8. SageMaker remote integration tests setup
   * Create IAM role with "SageMakerRole" and add required AWS Managed policies: AmazonSageMakerFullAccess
   * Change to the appropriate directory based on framework and job type of the image being tested.
   * Run pytest tests. The example below refers to testing mxnet_training images:
      ```shell
      pytest integration/sagemaker/test_mnist.py \
      --region us-west-2 --docker-base-name mxnet-training \
      --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
      --instance-type ml.g5.12xlarge
      ```

9. SageMaker benchmark tests setup
   * Create `sm_benchmark_env_settings.config` file
   * Add the following to the file (commented lines are optional):
        ```shell script
        export DLC_IMAGES="<image_uri_1-you-want-to-benchmark-test>"
        # export DLC_IMAGES="$DLC_IMAGES <image_uri_2-you-want-to-benchmark-test>"
        # export DLC_IMAGES="$DLC_IMAGES <image_uri_3-you-want-to-benchmark-test>"
        export BUILD_CONTEXT=PR
        export TEST_TYPE=benchmark-sagemaker
        export CODEBUILD_RESOLVED_SOURCE_VERSION=$USER
        export REGION=us-west-2
        ```
   * Run:
        ```shell script
        source sm_benchmark_env_settings.config
        ```
   * Run the tests for all images:
        ```shell
        pip install -r requirements.txt
        python test/testrunner.py
        ```
   * Run the tests for one individual framework image type:
        ```shell
        # Assuming that the cwd is deep-learning-containers/
        cd test/dlc_tests
        pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py