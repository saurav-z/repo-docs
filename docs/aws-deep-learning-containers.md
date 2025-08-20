# AWS Deep Learning Containers: Pre-built Docker Images for Machine Learning

**Get started quickly with optimized Docker images for training and serving your machine learning models on AWS.**

[View the original repository on GitHub](https://github.com/aws/deep-learning-containers)

## Key Features:

*   **Optimized Environments:** Pre-configured with popular deep learning frameworks like TensorFlow, PyTorch, and MXNet, along with NVIDIA CUDA (for GPU instances) and Intel MKL (for CPU instances).
*   **Easy Deployment:** Available in Amazon Elastic Container Registry (Amazon ECR), making them ideal for use with Amazon SageMaker, EC2, ECS, and EKS.
*   **Flexible Framework Support:** Supports multiple versions of TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Simplified Development:** Streamlined build and testing processes with clear instructions.

## Getting Started

This section provides a guide to building and testing AWS Deep Learning Containers on various platforms.

### Prerequisites

*   An active AWS account with appropriate permissions.  Ensure your AWS CLI is configured. Recommended IAM permissions include:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
*   Docker installed on your system (e.g., OSX, EC2).

### Build & Test Setup (Example: MXNet GPU Python3 Training)

1.  **Set Up Your Environment:**
    *   Clone the repository.
    *   Define environment variables (replace placeholders with your values):

    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```

2.  **Authenticate with ECR:**

    ```shell
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
    ```

3.  **Prepare Your Build Environment:**
    *   Create a Python virtual environment:

    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```

4.  **Initial Setup:**

    ```shell
    bash src/setup.sh mxnet
    ```

## Building Your Image

Dockerfiles follow a structured path.  Customize your build using the `buildspec.yml` files located in the framework-specific directories (e.g., `mxnet/training/buildspec.yml`).

1.  **Build All Images:**

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  **Build a Specific Image:**

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```

    *   `--image_types`: `training`/`inference`
    *   `--device_types`: `cpu`/`gpu`
    *   `--py_versions`: `py2`/`py3`

## Upgrading Framework Versions

To incorporate a new framework version (e.g., MXNet 1.7.0), modify the relevant `buildspec.yml` file and ensure the corresponding Dockerfile exists.

## Adding Artifacts & Packages

*   **Adding Artifacts:**  Use the `context` key in your `buildspec.yml` to include files from your build context.
*   **Adding Packages:** Modify your Dockerfile (e.g., the `Dockerfile.gpu` for the MXNet 1.6.0 py3 GPU image) to include the desired package using `pip install`.

## Running Tests Locally

Run tests locally for faster iteration using the pytest framework. Requires access to an AWS account.

1.  **Prerequisites**  Ensure the images you want to test are available locally.
2.  **Set Up Testing Environment:**
    ```shell
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    cd test/dlc_tests
    ```
3.  **Run Tests:**

    *   **All Tests (EC2):** `pytest -s -rA ec2/ -n=auto`
    *   **All Tests (ECS):** `pytest -s -rA ecs/ -n=auto`
    *   **All Tests (EKS)**
        ```bash
        cd ../
        export TEST_TYPE=eks
        python test/testrunner.py
        ```
    *   **Specific Test File:**  `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py`
    *   **Specific Test Function:** `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu`

4.  **SageMaker Local Mode Tests:** Launch an EC2 instance with a Deep Learning AMI and run the tests described in the original README.
5.  **SageMaker Remote Tests** : Run remote tests using SageMaker. Create an IAM role and run the pytest commands as detailed in the original README.
6.  **SageMaker Benchmark Tests** :  Perform benchmarking tests following the instructions in the original README.