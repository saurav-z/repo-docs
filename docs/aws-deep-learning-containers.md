# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started quickly with pre-built, optimized Docker images for training and deploying your machine learning models on AWS using [AWS Deep Learning Containers (DLCs)](https://github.com/aws/deep-learning-containers).**

## Key Features:

*   **Framework Support:** TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Optimized Environments:**  Includes Nvidia CUDA (for GPU instances) and Intel MKL (for CPU instances).
*   **Deployment Platforms:**  Tested on Amazon EC2, Amazon ECS, Amazon EKS, and default in Amazon SageMaker.
*   **Container Registry:** Available in Amazon Elastic Container Registry (Amazon ECR).
*   **Easy Integration:** Seamlessly integrates with AWS services for training, inference, and transformations.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building your Image](#building-your-image)
*   [Upgrading the framework version](#upgrading-the-framework-version)
*   [Adding artifacts to your build context](#adding-artifacts-to-your-build-context)
*   [Adding a package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

## Getting Started

This section provides instructions on building and testing the DLCs across various platforms, including Amazon SageMaker, EC2, ECS, and EKS. We will use a MXNet GPU python3 training container example.

**Prerequisites:**

*   An AWS account (configure your environment using the AWS CLI). We recommend an IAM role. The following managed permissions should suffice:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
*   An ECR repository (e.g., “beta-mxnet-training” in us-west-2).
*   Docker client.

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
3.  Create a virtual environment and install requirements:
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
4.  Perform initial setup:
    ```shell
    bash src/setup.sh mxnet
    ```

## Building your Image

Dockerfiles are located in paths following a specific pattern (e.g., `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`). Build specifications reside in `<framework>/<training|inference>/buildspec.yml` to specify the desired Dockerfile version.

1.  **Build all images defined in the buildspec.yml:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

2.  **Build a single image:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet --image_types training --device_types cpu --py_versions py3
    ```

3.  **Arguments:**
    *   `--image_types`:  `training` or `inference`.
    *   `--device_types`: `cpu` or `gpu`.
    *   `--py_versions`:  `py2` or `py3`.

4.  **Example: Build all GPU training containers:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet --image_types training --device_types gpu --py_versions py3
    ```

## Upgrading the framework version

To update a framework version (e.g., MXNet to 1.7.0):

1.  Modify the `buildspec.yml` file:

    ```yaml
    # mxnet/training/buildspec.yml
    # ...
    version: &VERSION 1.7.0  # Change to the new version
    # ...
    ```

2.  Ensure the Dockerfile exists (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).

3.  Build the container as described above.

## Adding artifacts to your build context

To include artifacts in your build context (e.g., copying `README-context.rst`):

1.  Add the artifact in the framework buildspec under the `context` key:

    ```yaml
    # mxnet/training/buildspec.yml
    context:
      README.xyz:  # Object name
        source: README-context.rst  # Source file
        target: README.rst        # Target file in build context
    ```

2.  You can use `training_context` or `inference_context` for training or inference images only.
3.  Alternatively, use the specific image context key.

4.  Build the container.

## Adding a package

To add a package to your image:

1.  Modify the Dockerfile:
    ```dockerfile
    # mxnet/training/docker/1.6.0/py3/Dockerfile.gpu
    RUN ${PIP} install --no-cache --upgrade \
        ...
        awscli \
        octopush  # Add the package
    ```

2.  Build the container.

## Running Tests Locally

Run tests locally to iterate and avoid resource consumption.

1.  Install testing requirements:

    ```shell
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  Set environment variables:

    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  Change the directory:

    ```shell
    cd test/dlc_tests
    ```
4.  Run tests:

    *   **All tests (series):**
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
    *   **Specific test file:**
        ```shell
        pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
        ```
    *   **Specific test function:**
        ```shell
        pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
        ```
5. **SageMaker local mode tests**
   * Clone your github branch with changes and run the following commands
       ```shell script
       git clone https://github.com/{github_account_id}/deep-learning-containers/
       cd deep-learning-containers && git checkout {branch_name}
       ```
   * Login into the ECR repo where the new docker images built exist
       ```shell script
       $(aws ecr get-login --no-include-email --registry-ids ${aws_id} --region ${aws_region})
       ```
   * Change to the appropriate directory (sagemaker_tests/{framework}/{job_type}) based on framework and job type of the image being tested.
       The example below refers to testing mxnet_training images
       ```shell script
       cd test/sagemaker_tests/mxnet/training/
       pip3 install -r requirements.txt
       ```
   * To run the SageMaker local integration tests (aside from tensorflow_inference), use the pytest command below:
       ```shell script
       python3 -m pytest -v integration/local --region us-west-2 \
       --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
        --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
        --py-version 3
       ```
   * To test tensorflow_inference py3 images, run the command below:
     ```shell script
     python3 -m  pytest -v integration/local \
     --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference \
     --tag 1.15.2-cpu-py36-ubuntu16.04 --framework-version 1.15.2 --processor cpu
     ```
6. **SageMaker remote tests setup**
   * Create an IAM role with name “SageMakerRole” and add the below AWS Managed policies
       ```
       AmazonSageMakerFullAccess
       ```
   *  Change to the appropriate directory (sagemaker_tests/{framework}/{job_type}) based on framework and job type of the image being tested."
       The example below refers to testing mxnet_training images
       ```shell script
       cd test/sagemaker_tests/mxnet/training/
       pip3 install -r requirements.txt
       ```
   *  To run the SageMaker remote integration tests (aside from tensorflow_inference), use the pytest command below:
       ```shell script
       pytest integration/sagemaker/test_mnist.py \
       --region us-west-2 --docker-base-name mxnet-training \
       --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
       --instance-type ml.g5.12xlarge
       ```
   * For tensorflow_inference py3 images run the below command
      ```shell script
      python3 -m pytest test/integration/sagemaker/test_tfs. --registry {aws_account_id} \
      --region us-west-2  --repo tensorflow-inference --instance-types ml.c5.18xlarge \
      --tag 1.15.2-py3-cpu-build --versions 1.15.2
      ```
7. **SageMaker benchmark tests setup**
    * Create a file named `sm_benchmark_env_settings.config` in the deep-learning-containers/ folder
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
    * To test all images for multiple frameworks, run:
        ```shell script
        pip install -r requirements.txt
        python test/testrunner.py
        ```
    * To test one individual framework image type, run:
        ```shell script
        # Assuming that the cwd is deep-learning-containers/
        cd test/dlc_tests
        pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py
        ```
    * The scripts and model-resources used in these tests will be located at:
        ```
        deep-learning-containers/test/dlc_tests/benchmark/sagemaker/<framework-name>/<image-type>/resources/
        ```

**[Back to the original repository](https://github.com/aws/deep-learning-containers)**