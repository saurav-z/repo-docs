# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

AWS Deep Learning Containers (DLCs) provide pre-built and optimized Docker images for training and serving your machine learning models, simplifying deployment across various platforms.  [Learn more on GitHub](https://github.com/aws/deep-learning-containers).

**Key Features:**

*   **Pre-built and Optimized:**  Includes TensorFlow, TensorFlow 2, PyTorch, and MXNet with optimized libraries (Nvidia CUDA for GPUs, Intel MKL for CPUs).
*   **Amazon ECR Availability:**  Images are readily available in Amazon Elastic Container Registry (ECR).
*   **SageMaker Integration:**  Seamlessly integrates with Amazon SageMaker for training, inference, and more.
*   **Platform Support:**  Tested and supported on Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Easy to Use:** Simplifies deployment and provides a consistent environment for machine learning workloads.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the framework version](#upgrading-the-framework-version)
*   [Adding artifacts to your build context](#adding-artifacts-to-your-build-context)
*   [Adding a package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

### Getting Started

This section details the setup to build and test DLCs on Amazon SageMaker, EC2, ECS, and EKS, using an example of building an MXNet GPU python3 training container.

**Prerequisites:**

*   An active AWS account with the following [IAM permissions](https://github.com/aws/deep-learning-containers#getting-started):
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess

*   An ECR repository (e.g., “beta-mxnet-training” in us-west-2).
*   Docker installed and configured on your system.

**Steps:**

1.  Clone the repository.
2.  Set environment variables:
    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
3.  Login to ECR:
    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
4.  Create and activate a virtual environment and install the required packages:
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
5.  Perform the initial setup:
    ```shell
    bash src/setup.sh mxnet
    ```

### Building Your Image

Build your custom Docker images using the provided `buildspec.yml` files that define the image configurations.

1.  To build all images from `buildspec.yml`:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

2.  To build a single image:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```

3.  Arguments `--image_types`, `--device_types`, and `--py_versions` accept comma-separated lists (e.g., `--image_types training,inference`).

### Upgrading the framework version

To upgrade the MXNet framework version (e.g., from 1.6.0 to 1.7.0):

1.  Modify the `buildspec.yml` file to update the `version` value for MXNet training.
2.  Ensure that the corresponding Dockerfile exists at the designated path (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  Build the container as described above.

### Adding artifacts to your build context

Add supporting files to your build context to be available inside of the docker containers during the build process.

1.  To copy a file, such as `README-context.rst`, into the build context, specify it in the `buildspec.yml` under the `context` key, which can be at the top-level, under `training_context`, or under the specific image.
2.  Specify the `source` and `target` file names.
3.  Build the container as described above.

### Adding a package

Modify the Dockerfile to install additional packages within your images.

1.  Edit the Dockerfile for the desired image (e.g., `mxnet/training/docker/1.6.0/py3/Dockerfile.gpu`).
2.  Add the package installation using the `RUN ${PIP} install` command.
3.  Build the container as described above.

### Running Tests Locally

Run tests locally to validate your changes and avoid consuming extra resources or waiting for builds.

1.  Set up a testing environment either on an EC2 instance or locally, ensuring the required images are pulled from ECR, and the required packages are installed:
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
3.  Navigate to the testing directory:
    ```shell
    cd test/dlc_tests
    ```
4.  Run tests for various platforms with pytest commands:
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
5.  Run specific test files or functions:
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

6.  Run SageMaker local mode tests with these configurations and commands:
    ```shell
   python3 -m pytest -v integration/local --region us-west-2 \
   --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
    --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
    --py-version 3
    ```
7.  For SageMaker remote tests, set up the IAM role and run the tests:
    ```shell
   pytest integration/sagemaker/test_mnist.py \
   --region us-west-2 --docker-base-name mxnet-training \
   --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
   --instance-type ml.g5.12xlarge
   ```
8.  SageMaker benchmark tests:
    *   Create a file named `sm_benchmark_env_settings.config` in the deep-learning-containers/ folder
    *   Add the following to the file (commented lines are optional):
        ```shell script
        export DLC_IMAGES="<image_uri_1-you-want-to-benchmark-test>"
        # export DLC_IMAGES="$DLC_IMAGES <image_uri_2-you-want-to-benchmark-test>"
        # export DLC_IMAGES="$DLC_IMAGES <image_uri_3-you-want-to-benchmark-test>"
        export BUILD_CONTEXT=PR
        export TEST_TYPE=benchmark-sagemaker
        export CODEBUILD_RESOLVED_SOURCE_VERSION=$USER
        export REGION=us-west-2
        ```
    *   Run:
        ```shell script
        source sm_benchmark_env_settings.config
        ```
    *   To test all images for multiple frameworks, run:
        ```shell script
        pip install -r requirements.txt
        python test/testrunner.py
        ```
    *   To test one individual framework image type, run:
        ```shell script
        # Assuming that the cwd is deep-learning-containers/
        cd test/dlc_tests
        pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py
        ```

Note: tensorflow\_inference py2 images are not supported with SageMaker.