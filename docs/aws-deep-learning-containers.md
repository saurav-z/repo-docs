# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started with pre-built, optimized Docker images for training and serving your machine learning models with AWS Deep Learning Containers (DLCs).** ([Original Repo](https://github.com/aws/deep-learning-containers))

AWS Deep Learning Containers (DLCs) provide pre-packaged environments for training and serving machine learning models, streamlining your development workflow. These Docker images are optimized for performance and ease of use, supporting popular frameworks like TensorFlow, PyTorch, and MXNet. They are readily available in Amazon Elastic Container Registry (Amazon ECR) and are seamlessly integrated with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.

**Key Features:**

*   **Optimized Environments:** Pre-configured with TensorFlow, TensorFlow 2, PyTorch, and MXNet, along with Nvidia CUDA (for GPU instances) and Intel MKL (for CPU instances).
*   **Pre-built Images:** Ready-to-use Docker images for various frameworks, versions, and configurations.
*   **Seamless Integration:** Designed for use with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Easy Deployment:** Simplifies the process of deploying and scaling machine learning models.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the Framework Version](#upgrading-the-framework-version)
*   [Adding Artifacts to Your Build Context](#adding-artifacts-to-your-build-context)
*   [Adding a Package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

## Getting Started

This section guides you through the setup process for building and testing DLCs on Amazon SageMaker, EC2, ECS, and EKS.

We'll use the example of building an *MXNet GPU python3 training* container.

**Prerequisites:**

*   Access to an AWS account. Configure your environment using the AWS CLI, either with an IAM user or, preferably, an IAM role. Recommended IAM permissions:

    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess

*   An [ECR repository](https://docs.aws.amazon.com/cli/latest/reference/ecr/create-repository.html) (e.g., "beta-mxnet-training" in us-west-2).
*   [Docker](https://docs.docker.com/get-docker/) installed and configured.

**Steps:**

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

3.  Create and activate a virtual environment and install the dependencies:

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

The paths to the Dockerfiles follow a specific pattern:  `\<framework>/<training|inference>/buildspec.yml`. You can customize the build process by modifying these files.

**Steps:**

1.  **Build all images defined in the `buildspec.yml`:**

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

3.  **Image type arguments:**

    *   `--image_types <training/inference>`
    *   `--device_types <cpu/gpu>`
    *   `--py_versions <py2/py3>`

4.  **Example: Build all GPU training containers:**

    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

## Upgrading the Framework Version

To upgrade the framework version:

1.  Modify the version in the buildspec.yml file (e.g., change `version: &VERSION 1.6.0` to `version: &VERSION 1.7.0`).

2.  Ensure the Dockerfile for the new version exists (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).

3.  Build the container as described above.

## Adding Artifacts to Your Build Context

To add artifacts to your build context:

1.  Add the artifact to the framework buildspec file under the `context` or `training_context` or `inference_context` key.

    ```yaml
    context:
      README.xyz:
        source: README-context.rst
        target: README.rst
    ```

2.  Build the container.

## Adding a Package

To add a package to your image:

1.  Modify the Dockerfile to include the package installation (e.g., add `octopush` to the `RUN pip install` command).

2.  Build the container.

## Running Tests Locally

Running tests locally is helpful for quick iterations and avoids resource consumption.

**Prerequisites:**

*   EC2 instance with the repo cloned or your local machine.
*   Docker images to be tested (pull from ECR).
*   Installed test requirements (install -r src/requirements.txt and pip install -r test/requirements.txt)

**Steps:**

1.  Change directories into the cloned folder.
    ```shell
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```

2.  Export environment variables.
    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

3.  Change directory into the test/dlc_tests folder.
    ```shell
    cd test/dlc_tests
    ```

4.  **Run tests (in series):**

    ```bash
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto
    #EKS
    cd ../
    export TEST_TYPE=eks
    python test/testrunner.py
    ```

5.  **Run a specific test file:**

    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

6.  **Run a specific test function:**

    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

7.  **SageMaker Local Mode Tests:**
    *   Launch EC2 instance with the latest Deep Learning AMI.
    *   Clone your github branch.
    *   Log into the ECR repo where the new docker images built exist.
    *   Change to the appropriate directory (sagemaker_tests/{framework}/{job_type}).
    *   Install test requirements
    *   To run the SageMaker local integration tests, use the pytest command as shown in the original README.

8.  **SageMaker Remote Tests:**
    *   Create an IAM role with SageMakerFullAccess.
    *   Change to the appropriate directory (sagemaker_tests/{framework}/{job_type}).
    *   To run the SageMaker remote integration tests, use the pytest command as shown in the original README.

9.  **SageMaker Benchmark Tests:**

    *   Create `sm_benchmark_env_settings.config` and add environment variables.
    *   Run `source sm_benchmark_env_settings.config`.
    *   Run tests using testrunner.py or run `pytest` commands to test one individual framework image type as shown in the original README.