# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started with pre-built, optimized Docker images for training and serving your machine learning models on AWS with [AWS Deep Learning Containers (DLCs)](https://github.com/aws/deep-learning-containers) and accelerate your ML workflows.**

**Key Features:**

*   **Pre-built & Optimized:** Ready-to-use Docker images with TensorFlow, TensorFlow 2, PyTorch, and MXNet, pre-configured with optimized libraries for CPU and GPU environments (Nvidia CUDA, Intel MKL).
*   **Seamless Integration:** Designed for use with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Easy Deployment:** Available in Amazon Elastic Container Registry (Amazon ECR), making deployment straightforward.
*   **Comprehensive Framework Support:** Covers popular deep learning frameworks and their versions.
*   **Flexible Testing:** Built-in support for local and remote testing with pytest.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the framework version](#upgrading-the-framework-version)
*   [Adding artifacts to your build context](#adding-artifacts-to-your-build-context)
*   [Adding a package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

## Getting Started

This section details the setup for building and testing DLCs on Amazon SageMaker, EC2, ECS, and EKS.

We'll demonstrate by building a ***MXNet GPU python3 training*** container.

**Prerequisites:**

*   Access to an AWS account with the necessary permissions.  Configure your environment using the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) (IAM role recommended).  Required IAM permissions include (but are not limited to):
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess

*   An ECR repository (e.g., "beta-mxnet-training" in us-west-2).
*   [Docker](https://docs.docker.com/get-docker/) installed.

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

3.  Create and activate a virtual environment and install requirements:
    ```bash
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```

4.  Perform the initial setup:
    ```bash
    bash src/setup.sh mxnet
    ```

## Building Your Image

The Dockerfiles follow a pattern:  `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`.  These paths are defined in the buildspec.yml file within each framework's directory (e.g., `mxnet/training/buildspec.yml`).  Modify `buildspec.yml` to specify the Dockerfile version.

1.  To build all images specified in `buildspec.yml` locally:
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

3.  Arguments for `--image_types`, `--device_types`, and `--py_versions` are comma-separated lists (see the original README for values).

4.  Example: Build all GPU training containers:
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```
## Upgrading the framework version
1. Suppose, if there is a new framework version for MXNet (version 1.7.0) then this would need to be changed in the
buildspec.yml file for MXNet training.
    ```yaml
    # mxnet/training/buildspec.yml
      1   account_id: &ACCOUNT_ID <set-$ACCOUNT_ID-in-environment>
      2   region: &REGION <set-$REGION-in-environment>
      3   framework: &FRAMEWORK mxnet
      4   version: &VERSION 1.6.0 *<--- Change this to 1.7.0*
          ................
    ```
2. The dockerfile for this should exist at mxnet/docker/1.7.0/py3/Dockerfile.gpu. This path is dictated by the
docker_file key for each repository.
    ```yaml
    # mxnet/training/buildspec.yml
     41   images:
     42     BuildMXNetCPUTrainPy3DockerImage:
     43       <<: *TRAINING_REPOSITORY
              ...................
     49       docker_file: !join [ docker/, *VERSION, /, *DOCKER_PYTHON_VERSION, /Dockerfile., *DEVICE_TYPE ]

    ```
3. Build the container as described above.

## Adding artifacts to your build context

1.  Add artifacts to the build context by specifying them in the framework's `buildspec.yml` under the `context` key.
    ```yaml
    # mxnet/training/buildspec.yml
     19 context:
     20   README.xyz: *<---- Object name (Can be anything)*
     21     source: README-context.rst *<--- Path for the file to be copied*
     22     target: README.rst *<--- Name for the object in** the build context*
    ```
2. To add the context to either training or inference images use `training_context` or `inference_context` respectively

## Adding a package

See [Building AWS Deep Learning Containers Custom Images](custom_images.md) for detailed customization instructions.
1.  Modify the Dockerfile (e.g., `mxnet/training/docker/1.6.0/py3/Dockerfile.gpu`) to include the package installation:

```dockerfile
    139 RUN ${PIP} install --no-cache --upgrade \
    140     keras-mxnet==2.2.4.2 \
    ...........................
    160     awscli \
    161     octopush
```
2.  Build the container as described above.

## Running Tests Locally

Run tests locally using pytest to avoid excessive resource usage.

1.  Make sure you have the images you want to test locally. Change directory into the cloned repository, install test requirements:
    ```bash
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  Set the `DLC_IMAGES`, `PYTHONPATH` and  `CODEBUILD_RESOLVED_SOURCE_VERSION` environment variables:
    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

3.  Navigate to the test directory:
    ```bash
    cd test/dlc_tests
    ```
4.  Run all tests for a given platform (EC2, ECS, EKS):
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
    Remove `-n=auto` for sequential execution.

5.  Run a specific test file:
    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

6.  Run a specific test function:
    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```
7.  SageMaker local mode testing instructions are available in the original README.
8.  SageMaker remote testing instructions are available in the original README.  You must create an IAM role and configure your account.
9.  SageMaker benchmark tests are detailed in the original README.  You will need to create a `sm_benchmark_env_settings.config` file.