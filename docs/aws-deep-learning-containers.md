# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Accelerate your machine learning workflows with AWS Deep Learning Containers (DLCs), pre-built Docker images optimized for training and serving models.** ([Original Repo](https://github.com/aws/deep-learning-containers))

## Key Features:

*   **Pre-built and Optimized:** Benefit from pre-configured environments with popular deep learning frameworks and libraries.
*   **Framework Support:**  Includes TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Hardware Acceleration:** Optimized for both CPU (with Intel MKL) and GPU (with Nvidia CUDA) instances.
*   **Integration:** Seamlessly integrates with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Available in ECR:** Easily access and deploy images from the Amazon Elastic Container Registry (ECR).

## Contents

*   [Getting Started](#getting-started)
*   [Building your Image](#building-your-image)
*   [Upgrading the framework version](#upgrading-the-framework-version)
*   [Adding artifacts to your build context](#adding-artifacts-to-your-build-context)
*   [Adding a package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

## Getting Started

This section provides instructions on how to set up your environment to build and test the DLCs on Amazon SageMaker, EC2, ECS, and EKS. This example focuses on building an MXNet GPU python3 training container.

1.  **Prerequisites:**
    *   An AWS account and configured AWS CLI access (IAM user or IAM role). Recommended IAM permissions: AmazonEC2ContainerRegistryFullAccess, AmazonEC2FullAccess, AmazonEKSClusterPolicy, AmazonEKSServicePolicy, AmazonEKSServiceRolePolicy, AWSServiceRoleForAmazonEKSNodegroup, AmazonSageMakerFullAccess, and AmazonS3FullAccess.
    *   An ECR repository (e.g., "beta-mxnet-training" in us-west-2).
    *   Docker installed and configured.
2.  **Setup:**
    *   Clone the repository and set environment variables:
        ```bash
        export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
        export REGION=us-west-2
        export REPOSITORY_NAME=beta-mxnet-training
        ```
    *   Login to ECR:
        ```bash
        aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
        ```
    *   Create and activate a virtual environment, then install requirements:
        ```bash
        python3 -m venv dlc
        source dlc/bin/activate
        pip install -r src/requirements.txt
        ```
    *   Perform initial setup for MXNet:
        ```bash
        bash src/setup.sh mxnet
        ```

## Building your Image

The image build process leverages `buildspec.yml` files, which specify the Dockerfile locations.

1.  **Build All Images:** Build all Dockerfiles specified in the `buildspec.yml` file:
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  **Build Specific Images:** Build a single image by specifying image type, device type, and Python version:
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
3.  **Arguments:** Use comma-separated lists for `--image_types`, `--device_types`, and `--py_versions`.  Valid values are:
    *   `--image_types`: `training`, `inference`
    *   `--device_types`: `cpu`, `gpu`
    *   `--py_versions`: `py2`, `py3`

## Upgrading the framework version

1.  **Modify buildspec.yml:** Update the framework version in the corresponding `buildspec.yml` file.  For example, to upgrade MXNet to version 1.7.0, change the `version` key in `mxnet/training/buildspec.yml`.
2.  **Dockerfile Location:** Ensure the Dockerfile for the new version exists at the expected path (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  **Build the Container:** Follow the instructions in "Building your Image" to build the updated container.

## Adding artifacts to your build context

1.  **Add Artifacts in buildspec.yml:** Add the artifact in the framework buildspec file under the context key:
   ```yaml
    # mxnet/training/buildspec.yml
     19 context:
     20   README.xyz: *<---- Object name (Can be anything)*
     21     source: README-context.rst *<--- Path for the file to be copied*
     22     target: README.rst *<--- Name for the object in** the build context*
   ```
2.  **Training/Inference Context:** If needed add it under `training_context` or `inference_context`.
3.  **Individual Container Context:** Add it under the `context` key for that particular image.
4.  **Build the Container:** Follow the instructions in "Building your Image" to build the updated container.

## Adding a package

1.  **Modify Dockerfile:**  Add the package installation command to the Dockerfile. For example, to add `octopush` to the MXNet 1.6.0 py3 GPU image:
    ```dockerfile
    139 RUN ${PIP} install --no-cache --upgrade \
    140     keras-mxnet==2.2.4.2 \
    ...........................
    160     awscli \
    161     octopush
    ```
2.  **Build the Container:** Follow the instructions in "Building your Image" to build the updated container.

## Running Tests Locally

Before running tests, make sure you have the images you want to test locally.

1.  **Setup:** Navigate to the cloned `deep-learning-containers/` directory and install requirements:
    ```bash
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Environment Variables:**  Set `DLC_IMAGES` (space-separated list of ECR URIs), `PYTHONPATH` (absolute path to `src/`), and  `CODEBUILD_RESOLVED_SOURCE_VERSION` (unique identifier).
    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Navigate to Test Directory:** Change directory to `test/dlc_tests`.
4.  **Run Tests:**
    *   **EC2:**  `pytest -s -rA ec2/ -n=auto`
    *   **ECS:**  `pytest -s -rA ecs/ -n=auto`
    *   **EKS:** `cd ../ && export TEST_TYPE=eks && python test/testrunner.py`
    *   **Specific Test File:** `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py`
    *   **Specific Test Function:** `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu`
5.  **SageMaker Local Tests:**
    *   Launch an EC2 instance with a Deep Learning AMI.
    *   Clone the repo and checkout your branch.
    *   Login to the ECR repo.
    *   Navigate to the appropriate directory (e.g., `test/sagemaker_tests/mxnet/training/`).
    *   Install requirements: `pip3 install -r requirements.txt`
    *   Run:
        ```bash
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
        --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
        --py-version 3
        ```
        or, for TensorFlow inference:
        ```bash
        python3 -m  pytest -v integration/local \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference \
        --tag 1.15.2-cpu-py36-ubuntu16.04 --framework-version 1.15.2 --processor cpu
        ```
6.  **SageMaker Remote Tests:**
    *   Create an IAM role named "SageMakerRole" with  `AmazonSageMakerFullAccess` permissions.
    *   Navigate to the appropriate test directory.
    *   Install requirements.
    *   Run:
        ```bash
        pytest integration/sagemaker/test_mnist.py \
        --region us-west-2 --docker-base-name mxnet-training \
        --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
        --instance-type ml.g5.12xlarge
        ```
       or, for TensorFlow inference:
       ```bash
       python3 -m pytest test/integration/sagemaker/test_tfs. --registry {aws_account_id} \
       --region us-west-2  --repo tensorflow-inference --instance-types ml.c5.18xlarge \
       --tag 1.15.2-py3-cpu-build --versions 1.15.2
       ```
7.  **SageMaker Benchmark Tests:**
    *   Create `sm_benchmark_env_settings.config` with:
        ```bash
        export DLC_IMAGES="<image_uri_1-you-want-to-benchmark-test>"
        export BUILD_CONTEXT=PR
        export TEST_TYPE=benchmark-sagemaker
        export CODEBUILD_RESOLVED_SOURCE_VERSION=$USER
        export REGION=us-west-2
        ```
    *   `source sm_benchmark_env_settings.config`
    *   Run: `pip install -r requirements.txt && python test/testrunner.py`
    *   Or, for a specific framework:
        ```bash
        cd test/dlc_tests
        pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py
        ```