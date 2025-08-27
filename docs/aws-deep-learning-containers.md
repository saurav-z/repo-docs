# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started quickly with pre-built, optimized Docker images for training and deploying your machine learning models.** [Learn more about DLCs](https://github.com/aws/deep-learning-containers)

AWS Deep Learning Containers (DLCs) are a collection of Docker images designed to streamline your machine learning workflows. These containers provide pre-configured environments with popular deep learning frameworks and essential libraries, making it easier to train and deploy your models on various AWS services.

## Key Features:

*   **Optimized Environments:** Pre-built with TensorFlow, TensorFlow 2, PyTorch, and MXNet, including optimized libraries for CPU (Intel MKL) and GPU (Nvidia CUDA) instances.
*   **Seamless Integration:** Designed for use with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Easy Deployment:** Available in Amazon Elastic Container Registry (Amazon ECR) for easy access and deployment.
*   **Framework Support:** Supports a wide range of frameworks, including TensorFlow, PyTorch, and MXNet.

## Table of Contents:

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the Framework Version](#upgrading-the-framework-version)
*   [Adding Artifacts to Your Build Context](#adding-artifacts-to-your-build-context)
*   [Adding a Package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

### Getting Started

This section outlines the steps to set up your environment to build and test DLCs on Amazon SageMaker, EC2, ECS, and EKS.

**Example: Building a MXNet GPU python3 training container:**

1.  **Prerequisites:**
    *   Access to an AWS account with appropriate permissions (IAM role recommended; see original README for managed policy suggestions).
    *   An ECR repository (e.g., "beta-mxnet-training" in us-west-2).
    *   Docker client installed on your system.

2.  **Configuration:**
    *   Clone the repository.
    *   Set environment variables:
        ```bash
        export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
        export REGION=us-west-2
        export REPOSITORY_NAME=beta-mxnet-training
        ```
    *   Login to ECR:
        ```bash
        aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
        ```
    *   Create and activate a virtual environment and install requirements:
        ```bash
        python3 -m venv dlc
        source dlc/bin/activate
        pip install -r src/requirements.txt
        ```
    *   Perform initial setup:
        ```bash
        bash src/setup.sh mxnet
        ```

### Building Your Image

DLCs utilize `buildspec.yml` files to define the image build process. These files specify the Dockerfile paths and other build configurations.

1.  **Building all images:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

2.  **Building a specific image:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```

3.  **Image Type, Device Type and Python Version arguments:**
    *   `--image_types`: `<training/inference>`
    *   `--device_types`: `<cpu/gpu>`
    *   `--py_versions`: `<py2/py3>`

### Upgrading the Framework Version

To upgrade the framework version:

1.  Modify the `version` key in the `buildspec.yml` file (e.g., change MXNet version from 1.6.0 to 1.7.0).
2.  Ensure the corresponding Dockerfile exists at the specified path (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  Build the container as described above.

### Adding Artifacts to Your Build Context

To include artifacts (files) in the build context:

1.  Add the artifact information under the `context` key in the buildspec file:
    ```yaml
    context:
      README.xyz:
        source: README-context.rst
        target: README.rst
    ```
2.  Artifacts can also be added to `training_context` or `inference_context` or within a specific image's context.
3.  Build the container as described above.

### Adding a Package

To add a package to your image:

1.  Modify the Dockerfile to include the `pip install` command for the new package. For example, adding `octopush`:

    ```dockerfile
    RUN ${PIP} install --no-cache --upgrade \
        ...
        awscli \
        octopush
    ```
2.  Build the container as described above.

### Running Tests Locally

This section provides instructions on how to run tests locally, using the pytest framework.  Requires an AWS account.

1.  **Prerequisites:**  Ensure you have the necessary images (pull them from ECR if needed), and install the requirements.
    ```bash
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```

2.  **Environment Variables:**
    *   Set `DLC_IMAGES` to a space-separated list of ECR URIs to be tested.
    *   Set `CODEBUILD_RESOLVED_SOURCE_VERSION` to a unique identifier.
    *   Set `PYTHONPATH` to the absolute path of the `src/` folder.

    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

3.  **Change Directories:**
    ```bash
    cd test/dlc_tests
    ```

4.  **Run all tests (in series) for a given platform:**
    ```bash
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto
    # EKS
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
    *   Launch a CPU or GPU EC2 instance with the latest Deep Learning AMI.
    *   Clone your github branch with changes and run the following commands.
        ```bash
        git clone https://github.com/{github_account_id}/deep-learning-containers/
        cd deep-learning-containers && git checkout {branch_name}
        ```
    *   Login into the ECR repo where the new docker images built exist
        ```bash
        $(aws ecr get-login --no-include-email --registry-ids ${aws_id} --region ${aws_region})
        ```
    *   Change to the appropriate directory based on framework and job type of the image being tested.
        ```bash
        cd test/sagemaker_tests/mxnet/training/
        pip3 install -r requirements.txt
        ```
    *   To run the SageMaker local integration tests (aside from tensorflow_inference), use the pytest command below:
        ```bash
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
         --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
         --py-version 3
        ```
    *   To test tensorflow_inference py3 images, run the command below:
        ```bash
        python3 -m  pytest -v integration/local \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference \
        --tag 1.15.2-cpu-py36-ubuntu16.04 --framework-version 1.15.2 --processor cpu
        ```
8.  **SageMaker Remote Tests on your account:**
    *   Create an IAM role named "SageMakerRole" with "AmazonSageMakerFullAccess" managed policy.
    *   Change to the appropriate directory:
        ```bash
        cd test/sagemaker_tests/mxnet/training/
        pip3 install -r requirements.txt
        ```
    *   To run the SageMaker remote integration tests (aside from tensorflow_inference), use the pytest command below:
        ```bash
        pytest integration/sagemaker/test_mnist.py \
        --region us-west-2 --docker-base-name mxnet-training \
        --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
        --instance-type ml.g5.12xlarge
        ```
    *   For tensorflow_inference py3 images, run the below command:
        ```bash
        python3 -m pytest test/integration/sagemaker/test_tfs. --registry {aws_account_id} \
        --region us-west-2  --repo tensorflow-inference --instance-types ml.c5.18xlarge \
        --tag 1.15.2-py3-cpu-build --versions 1.15.2
        ```

9.  **SageMaker Benchmark Tests:**
    *   Create a file named `sm_benchmark_env_settings.config`.
    *   Add the following to the file:
        ```bash
        export DLC_IMAGES="<image_uri_1-you-want-to-benchmark-test>"
        # export DLC_IMAGES="$DLC_IMAGES <image_uri_2-you-want-to-benchmark-test>"
        # export DLC_IMAGES="$DLC_IMAGES <image_uri_3-you-want-to-benchmark-test>"
        export BUILD_CONTEXT=PR
        export TEST_TYPE=benchmark-sagemaker
        export CODEBUILD_RESOLVED_SOURCE_VERSION=$USER
        export REGION=us-west-2
        ```
    *   Run: `source sm_benchmark_env_settings.config`
    *   To test all images for multiple frameworks, run:
        ```bash
        pip install -r requirements.txt
        python test/testrunner.py
        ```
    *   To test one individual framework image type, run:
        ```bash
        # Assuming that the cwd is deep-learning-containers/
        cd test/dlc_tests
        pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py
        ```
    *   Benchmark test resources are located at:
        ```
        deep-learning-containers/test/dlc_tests/benchmark/sagemaker/<framework-name>/<image-type>/resources/
        ```

**Note:** SageMaker does not support tensorflow_inference py2 images.