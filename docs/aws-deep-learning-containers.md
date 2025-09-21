# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get pre-built, optimized Docker images from AWS for training and serving your machine learning models using popular frameworks like TensorFlow, PyTorch, and MXNet.** [View the original repository](https://github.com/aws/deep-learning-containers).

## Key Features

*   **Optimized Environments:** Pre-configured with TensorFlow, TensorFlow 2, PyTorch, MXNet, Nvidia CUDA (for GPU instances), and Intel MKL (for CPU instances).
*   **Framework Support:** Supports a wide range of deep learning frameworks including TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Integration with AWS Services:** Seamlessly integrates with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Available on Amazon ECR:** Images are readily available in the Amazon Elastic Container Registry (Amazon ECR).

## Getting Started

This section provides instructions for building, testing, and using the AWS Deep Learning Containers (DLCs) on platforms like Amazon SageMaker, EC2, ECS, and EKS.

### Prerequisites

1.  **AWS Account & Permissions:** Ensure you have access to an AWS account and have configured your environment using the AWS CLI. Recommended IAM role permissions:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
2.  **ECR Repository:** Create an ECR repository (e.g., "beta-mxnet-training") in your desired AWS region.
3.  **Docker:** Ensure Docker is installed and configured on your system.

### Building and Testing

1.  **Clone the Repository:** Clone the AWS Deep Learning Containers repository.
2.  **Set Environment Variables:**  Set environment variables: `ACCOUNT_ID`, `REGION`, `REPOSITORY_NAME`.
    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
3.  **Login to ECR:** Authenticate with ECR.
    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
4.  **Create Virtual Environment & Install Requirements:**  Set up a Python virtual environment and install dependencies.
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
5.  **Initial Setup:** Perform initial setup for the container.
    ```shell
    bash src/setup.sh mxnet
    ```

### Building Images

*   Dockerfiles are located based on the framework, training/inference type, version, Python version, and processor.
*   Modify the buildspec.yml file to build dockerfiles.

1.  **Build All Images (from buildspec.yml):**
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

### Upgrading Framework Version

1.  Update the `version` key in the appropriate `buildspec.yml` file (e.g., `mxnet/training/buildspec.yml`).
2.  The corresponding Dockerfile path needs to be created/exist.
3.  Build the container as described above.

### Adding Artifacts to the Build Context

1.  Add files to the build context in the buildspec.yml under the `context`, `training_context` or `inference_context` key.
    ```yaml
    context:
        README.xyz:
            source: README-context.rst
            target: README.rst
    ```
2.  Build the container as described above.

### Adding a Package

1.  Modify the Dockerfile to include the `RUN pip install` command for the desired package.
    ```dockerfile
    RUN ${PIP} install --no-cache --upgrade \
        octopush
    ```
2.  Build the container as described above.

### Running Tests Locally

1.  **Install Test Requirements:** Install the testing dependencies.
    ```shell
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Set Environment Variables:**  Define environment variables. Replace `ACCOUNT_ID` with your account ID and specify image URIs.
    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Change Directory:** Navigate to the test directory.
    ```shell
    cd test/dlc_tests
    ```
4.  **Run Tests:**

    *   **EC2:** `pytest -s -rA ec2/ -n=auto`
    *   **ECS:** `pytest -s -rA ecs/ -n=auto`
    *   **EKS:** `cd ../; export TEST_TYPE=eks; python test/testrunner.py`
    *   **Specific Test File:** `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py`
    *   **Specific Test Function:** `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu`

### SageMaker Local/Remote Tests

Follow the instructions within the original README for running tests in SageMaker local mode and remote modes.
*   Configure prerequisites like IAM Roles and requirements.
*   Run the appropriate pytest commands.

### SageMaker Benchmark Tests

Follow the instructions within the original README for running the Benchmark tests.
*   Create and configure a `sm_benchmark_env_settings.config` file.
*   Run the appropriate commands