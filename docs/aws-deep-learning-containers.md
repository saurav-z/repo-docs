# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started quickly with pre-built, optimized Docker images for training and deploying your machine learning models on AWS.** [Explore the original repository](https://github.com/aws/deep-learning-containers).

## Key Features:

*   **Framework Support:** Built-in support for popular deep learning frameworks like TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Optimized Environments:** Pre-configured with Nvidia CUDA (for GPU instances) and Intel MKL (for CPU instances) libraries for peak performance.
*   **Amazon ECR Availability:** Easy access to images via Amazon Elastic Container Registry (ECR).
*   **SageMaker Integration:** Designed to work seamlessly with Amazon SageMaker for training, inference, and more.
*   **Platform Compatibility:** Tested and validated for use on Amazon EC2, Amazon ECS, and Amazon EKS.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the framework version](#upgrading-the-framework-version)
*   [Adding artifacts to your build context](#adding-artifacts-to-your-build-context)
*   [Adding a package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

### Getting Started

This section provides instructions to build and test Deep Learning Containers (DLCs) on Amazon SageMaker, EC2, ECS, and EKS.

We will use the ***MXNet GPU python3 training*** container example.

**Prerequisites:**

*   An active AWS account. [Configure your AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) for access. We recommend using an IAM role. The following managed permissions should suffice:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
*   An [ECR repository](https://docs.aws.amazon.com/cli/latest/reference/ecr/create-repository.html) named “beta-mxnet-training” in the us-west-2 region (or your preferred region).
*   [Docker](https://docs.docker.com/get-docker/) installed and configured on your system.

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

4.  Perform the initial setup:

    ```shell
    bash src/setup.sh mxnet
    ```

### Building Your Image

Dockerfiles follow a specific pattern: `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`. These paths are specified in the `buildspec.yml` files (e.g., `mxnet/training/buildspec.yml`). To build a specific image version, modify the buildspec file accordingly.

1.  To build all Dockerfiles defined in the buildspec.yml locally:

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

3.  The arguments `--image_types`, `--device_types`, and `--py_versions` accept comma-separated lists:

    *   `--image_types`: `<training/inference>`
    *   `--device_types`: `<cpu/gpu>`
    *   `--py_versions`: `<py2/py3>`

4.  Example: Build all GPU training containers:

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

### Upgrading the framework version

1.  To update to a new MXNet version (e.g., 1.7.0), modify the `buildspec.yml` file:

    ```yaml
    # mxnet/training/buildspec.yml
    # ...
    version: &VERSION 1.7.0 # Change to 1.7.0
    # ...
    ```

2.  Ensure the corresponding Dockerfile exists (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`). This is specified by the `docker_file` key in the buildspec.

    ```yaml
    # mxnet/training/buildspec.yml
    # ...
    docker_file: !join [ docker/, *VERSION, /, *DOCKER_PYTHON_VERSION, /Dockerfile., *DEVICE_TYPE ]
    # ...
    ```

3.  Build the container as described above.

### Adding artifacts to your build context

1.  To copy an artifact into the build context, add it to the framework's buildspec file under the `context` key. For example:

    ```yaml
    # mxnet/training/buildspec.yml
    context:
      README.xyz: # Object name (Can be anything)
        source: README-context.rst # Path for the file to be copied
        target: README.rst # Name for the object in the build context
    ```

2.  Use `training_context` or `inference_context` to make it available only for training or inference images.
3.  For a single container, add it under the context key for that particular image.
4.  Build the container as described above.

### Adding a package

1.  To add a package to your image (e.g., to the MXNet 1.6.0 py3 GPU image), modify the Dockerfile:

    ```dockerfile
    # mxnet/training/docker/1.6.0/py3/Dockerfile.gpu
    # ...
    RUN ${PIP} install --no-cache --upgrade \
        # ...
        awscli \
        octopush  # Add the package
    ```

2.  Build the container as described above.  For more information on customizing your container, see [Building AWS Deep Learning Containers Custom Images](custom_images.md).

### Running Tests Locally

Run tests locally to speed up the development process.

1.  Ensure you have the desired images. Pull them from ECR if necessary. Change the directory in your shell to cloned folder, then install requirements:

    ```shell
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```

2.  Set environment variables.  Modify the repository names to match your setup.

    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

3.  Navigate to the test directory:

    ```shell
    cd test/dlc_tests
    ```

4.  Run all tests for a given platform (in series):

    ```shell
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto
    ```
    ```shell
    #EKS
    cd ../
    export TEST_TYPE=eks
    python test/testrunner.py
    ```
    Remove `-n=auto` to run the tests sequentially.

5.  Run a specific test file:

    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

6.  Run a specific test function:

    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

7.  SageMaker Local Mode Testing:

    *   Launch an EC2 instance with the latest Deep Learning AMI.
    *   Clone the repo and install dependencies:
        ```shell
        git clone https://github.com/{github_account_id}/deep-learning-containers/
        cd deep-learning-containers && git checkout {branch_name}
        $(aws ecr get-login --no-include-email --registry-ids ${aws_id} --region ${aws_region}) # login into ECR repo
        cd test/sagemaker_tests/mxnet/training/
        pip3 install -r requirements.txt
        ```
    *   Run local SageMaker tests:
        ```shell
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
         --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
         --py-version 3
        ```
        For tensorflow_inference py3 images:
          ```shell
          python3 -m  pytest -v integration/local \
          --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference \
          --tag 1.15.2-cpu-py36-ubuntu16.04 --framework-version 1.15.2 --processor cpu
          ```
8.  SageMaker Remote Tests:

    *   Create an IAM role named “SageMakerRole” with the AmazonSageMakerFullAccess policy.
    *   Run the tests:
        ```shell
        cd test/sagemaker_tests/mxnet/training/
        pip3 install -r requirements.txt
        pytest integration/sagemaker/test_mnist.py \
        --region us-west-2 --docker-base-name mxnet-training \
        --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
        --instance-type ml.g5.12xlarge
        ```
        For tensorflow_inference py3 images:
        ```shell
        python3 -m pytest test/integration/sagemaker/test_tfs. --registry {aws_account_id} \
        --region us-west-2  --repo tensorflow-inference --instance-types ml.c5.18xlarge \
        --tag 1.15.2-py3-cpu-build --versions 1.15.2
        ```
9.  SageMaker Benchmark Tests:

    *   Create a file named `sm_benchmark_env_settings.config` in the root directory.
    *   Populate it with the following (modify as needed):
        ```shell
        export DLC_IMAGES="<image_uri_1-you-want-to-benchmark-test>"
        # export DLC_IMAGES="$DLC_IMAGES <image_uri_2-you-want-to-benchmark-test>"
        # export DLC_IMAGES="$DLC_IMAGES <image_uri_3-you-want-to-benchmark-test>"
        export BUILD_CONTEXT=PR
        export TEST_TYPE=benchmark-sagemaker
        export CODEBUILD_RESOLVED_SOURCE_VERSION=$USER
        export REGION=us-west-2
        ```
    *   Source the settings:
        ```shell
        source sm_benchmark_env_settings.config
        ```
    *   Run all tests:
        ```shell
        pip install -r requirements.txt
        python test/testrunner.py
        ```
    *   Run a specific framework test:
        ```shell
        cd test/dlc_tests
        pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py
        ```
    *   Benchmark test resources are located in:
        `deep-learning-containers/test/dlc_tests/benchmark/sagemaker/<framework-name>/<image-type>/resources/`

Note: SageMaker does not support tensorflow_inference py2 images.