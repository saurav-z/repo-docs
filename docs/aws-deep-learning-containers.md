# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started with pre-built, optimized Docker images for training and serving machine learning models with AWS Deep Learning Containers, and explore the original repository [here](https://github.com/aws/deep-learning-containers).**

## Key Features:

*   **Optimized Environments:** Pre-configured with popular deep learning frameworks like TensorFlow, PyTorch, and MXNet, as well as CUDA and Intel MKL libraries for optimal performance on both CPU and GPU instances.
*   **Framework Support:**  Includes TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Amazon Integration:**  Seamlessly integrates with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS for easy deployment and scaling of your machine learning workloads.
*   **Pre-built Images:**  Available in Amazon Elastic Container Registry (Amazon ECR), reducing the time and effort needed for environment setup.
*   **Customization:** Easily build and customize your own container images by modifying the provided Dockerfiles.
*   **Testing Framework:** Provides a pytest-based testing framework for local and remote testing on EC2, ECS, EKS, and SageMaker.

## Getting Started

This section provides instructions for building and testing AWS Deep Learning Containers.

### Prerequisites
*   **AWS Account:** Ensure you have an active AWS account with the necessary permissions.  Recommended permissions include:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
*   **AWS CLI:**  Configure the AWS Command Line Interface (CLI) to access your account.
*   **Docker:** Ensure you have Docker installed and configured on your system.

### Building an Example Container

The following steps demonstrate how to build an example MXNet GPU Python 3 training container.

1.  **Set up:**
    *   Clone the repository.
    *   Set the following environment variables, replacing the placeholder values with your account-specific information:
        ```bash
        export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
        export REGION=us-west-2
        export REPOSITORY_NAME=beta-mxnet-training
        ```
    *   Login to ECR:
        ```bash
        aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
        ```
    *   Create and activate a virtual environment and install dependencies:
        ```bash
        python3 -m venv dlc
        source dlc/bin/activate
        pip install -r src/requirements.txt
        ```
    *   Perform the initial setup:
        ```bash
        bash src/setup.sh mxnet
        ```

### Building Your Image

1.  **Locate Dockerfiles:**  The Dockerfiles follow a specific directory structure, such as `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`.
2.  **Build all images:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
3.  **Build a single image:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
4.  **Image Type, Device Type, and Python Version:** The arguments are comma-separated lists, with possible values as follows:
    ```bash
    --image_types <training/inference>
    --device_types <cpu/gpu>
    --py_versions <py2/py3>
    ```

### Upgrading the Framework Version

To upgrade the framework version, follow these steps:

1.  **Modify `buildspec.yml`:** Update the framework version in the `buildspec.yml` file (e.g., `mxnet/training/buildspec.yml`).
2.  **Locate the Dockerfile:** Ensure a Dockerfile exists for the new version (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  **Build the Container:** Build the container as described above.

### Adding Artifacts to Your Build Context

To copy artifacts from your build context:

1.  **Specify Context in `buildspec.yml`:** Add the artifact and its source and target paths within the `context`, `training_context`, or `inference_context` sections of the buildspec file.
2.  **Build the Container:** Build the container as described above.

### Adding a Package

To add a package to your image:

1.  **Modify the Dockerfile:** Update the Dockerfile to include the new package installation command (e.g., using `pip install`).
2.  **Build the Container:** Build the container as described above.

### Running Tests Locally

To run tests locally:

1.  **Prepare:** Install test requirements and ensure you have the necessary images.
2.  **Set Environment Variables:**  Set the `DLC_IMAGES`, `PYTHONPATH`, and `CODEBUILD_RESOLVED_SOURCE_VERSION` environment variables.
3.  **Navigate to Test Directory:** Change directories to `test/dlc_tests`.
4.  **Run Tests:**
    *   For EC2: `pytest -s -rA ec2/ -n=auto`
    *   For ECS: `pytest -s -rA ecs/ -n=auto`
    *   For EKS:
        ```bash
        cd ../
        export TEST_TYPE=eks
        python test/testrunner.py
        ```
    *   To run a specific test file: `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py`
    *   To run a specific test function:  `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu`
5.  **SageMaker Local Mode Tests:**  Run the provided pytest commands within a Deep Learning AMI EC2 instance.
6.  **SageMaker Remote Tests:** Configure an IAM role and run the appropriate pytest commands for testing in SageMaker.
7.  **SageMaker Benchmark Tests:** Set up the `sm_benchmark_env_settings.config` file and run the test runner.