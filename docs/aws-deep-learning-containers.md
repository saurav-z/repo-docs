# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Accelerate your machine learning projects with AWS Deep Learning Containers, pre-built and optimized Docker images for popular deep learning frameworks.**

[Link to Original Repo: AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers)

**Key Features:**

*   **Pre-built and Optimized:** Ready-to-use Docker images with TensorFlow, TensorFlow 2, PyTorch, and MXNet, pre-configured with necessary libraries like Nvidia CUDA (for GPU instances) and Intel MKL (for CPU instances).
*   **Amazon ECR Availability:** Easily accessible and deployable from the Amazon Elastic Container Registry (Amazon ECR).
*   **SageMaker Integration:** Seamlessly integrated as the default framework for Amazon SageMaker jobs (training, inference, transforms, etc.).
*   **EC2, ECS, and EKS Compatibility:** Tested and validated for use with Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Framework Support:** TensorFlow, TensorFlow 2, PyTorch, and MXNet.

## Getting Started

This section provides instructions on how to build and test the DLCs on various platforms (Amazon SageMaker, EC2, ECS, and EKS). The following instructions will guide you through building a MXNet GPU python3 training container:

**Prerequisites:**

*   An AWS account with appropriate permissions (see [Getting Started](#getting-started) in the original README for specific IAM policy recommendations).
*   Docker installed on your system.

**Setup:**

1.  **Configure AWS CLI:**  Set up your AWS CLI to access your account.
2.  **Create ECR Repository:** Create an ECR repository (e.g., "beta-mxnet-training") in your preferred region (e.g., us-west-2).
3.  **Clone the Repository:** Clone the AWS Deep Learning Containers repository.
4.  **Set Environment Variables:**  Set environment variables for your account ID, region, and repository name.
    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
5.  **Login to ECR:** Login to ECR using the AWS CLI.
    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
6.  **Create and Activate Virtual Environment:** Create a virtual environment and install dependencies.
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
7.  **Initial Setup:** Run the initial setup script.
    ```shell
    bash src/setup.sh mxnet
    ```

## Building Your Image

Build your Docker images using the provided build scripts. The paths to the Dockerfiles follow a specific pattern and are defined in the `buildspec.yml` files located in the framework directories.

1.  **Build All Images:** Build all images specified in the `buildspec.yml` file using the following command:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  **Build a Single Image:** Build a specific image by specifying image type, device type, and Python version.
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
3.  **Arguments:** The `--image_types`, `--device_types`, and `--py_versions` arguments accept comma-separated lists:
    ```
    --image_types <training/inference>
    --device_types <cpu/gpu>
    --py_versions <py2/py3>
    ```

## Upgrading the Framework Version

To upgrade the framework version:

1.  **Modify `buildspec.yml`:**  Update the `version` key in the relevant `buildspec.yml` file.
2.  **Docker File Location:** Ensure the Dockerfile for the new version exists at the correct path (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  **Build the Container:** Build the container using the instructions above.

## Adding Artifacts to Your Build Context

Include additional files in your build context using the `context` key in the `buildspec.yml` file:

1.  **Add Files to Context:** In your `buildspec.yml`, add the context of the files under the `context` key
2.  **Image Level Context:** Context can be defined for all images, specific images or image types.

## Adding a Package

To add a package to your image, modify the Dockerfile:

1.  **Modify Dockerfile:** Add the `RUN ${PIP} install` command with the package name.
2.  **Build the Container:** Build the container using the instructions above.

## Running Tests Locally

Run tests locally to validate your changes.

1.  **Prerequisites:** Ensure you have the images to test locally (pull them from ECR), install the required dependencies.
    ```shell
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```

2.  **Set Environment Variables:** Set environment variables including the paths to the images to test.
    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Navigate to Test Directory:** Change your directory into the test folder.
    ```shell
    cd test/dlc_tests
    ```
4.  **Run Tests:** Run tests using `pytest`.  Use the commands below to run tests in EC2, ECS, EKS, SageMaker.
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
5.  **Running Specific Tests:**  Specify test files or functions to run specific tests.

## SageMaker Testing
Follow the commands in the original README to run specific SageMaker local and remote tests with Pytest.