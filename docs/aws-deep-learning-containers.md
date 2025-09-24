# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started quickly with pre-built, optimized Docker images for training and serving your machine learning models with [AWS Deep Learning Containers (DLCs)](https://github.com/aws/deep-learning-containers), built for efficiency and performance.**

## Key Features:

*   **Pre-built and Optimized:** Includes TensorFlow, TensorFlow 2, PyTorch, and MXNet, with optimized libraries like NVIDIA CUDA (GPU) and Intel MKL (CPU).
*   **Easy Integration:** Works seamlessly with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Broad Availability:** Available in Amazon Elastic Container Registry (Amazon ECR).
*   **Flexibility:** Supports various frameworks and versions, allowing you to tailor your environment.
*   **Customization:** Provides guidance on building custom images and adding packages.
*   **Testing Support:** Comprehensive testing framework using pytest for local and remote testing.

## Getting Started

This section provides instructions for building and testing the DLCs on various AWS platforms like Amazon SageMaker, EC2, ECS, and EKS.  We will use an example of building an MXNet GPU python3 training container.

### Prerequisites

1.  **AWS Account Setup:**  Configure your AWS environment with the AWS CLI, ensuring access through an IAM user or role. Recommended managed permissions include:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
2.  **Create an ECR Repository:**  Create an ECR repository (e.g., "beta-mxnet-training") in your desired AWS region (e.g., us-west-2).
3.  **Docker Installation:** Ensure Docker is installed and set up on your system (OSX/EC2).

### Build and Test Steps

1.  **Clone the Repository:** Clone the AWS Deep Learning Containers repository.
2.  **Set Environment Variables:** Set the following environment variables, replacing placeholders with your actual values:

    ```shell script
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
3.  **Log in to ECR:** Authenticate with your ECR repository using the Docker login command:

    ```shell script
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
4.  **Create and Activate a Virtual Environment:** Create a virtual environment, activate it, and install the required packages.

    ```shell script
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
5.  **Initial Setup:** Run the setup script for your chosen framework (e.g., MXNet).

    ```shell script
    bash src/setup.sh mxnet
    ```

## Building Your Image

The Dockerfiles are located under specific paths (e.g., `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`), as defined by the `buildspec.yml` files within each framework's directory.

1.  **Build all images:** Build all images defined in the `buildspec.yml` file using the following command:

    ```shell script
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

2.  **Build a single image:** You can build a specific image by specifying the image type, device type, and Python version:

    ```shell script
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```

    Use the following arguments for filtering:
    *   `--image_types <training/inference>`
    *   `--device_types <cpu/gpu>`
    *   `--py_versions <py2/py3>`

3.  **Upgrading Framework Version:** To upgrade the framework version, modify the `version` key in the relevant `buildspec.yml` file and ensure the corresponding Dockerfile exists.

4.  **Adding Artifacts to your Build Context:** To include files/artifacts (like configuration files, scripts, or data) in your build context, define them in the buildspec.yml file's context section.  You can specify files at the top-level, or within training or inference contexts.

5.  **Adding a Package:** Customize your image by adding packages directly in the Dockerfile, as demonstrated in the original documentation.

## Running Tests Locally

The project uses `pytest` for testing.

1.  **Setup:**  Ensure you have the necessary images, and install test requirements.
    ```shell script
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Set Environment Variables:**  Set environment variables to specify the images to test, the Python path, and a unique identifier for resources.

    ```shell script
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Navigate to Test Directory:** Change directories into the test directory.

    ```shell script
    cd test/dlc_tests
    ```
4.  **Run Tests:**  Run all tests for a specific platform:

    ```shell script
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto

    #EKS
    cd ../
    export TEST_TYPE=eks
    python test/testrunner.py
    ```
5.  **Run Specific Tests:** Run specific test files or test functions using the appropriate `pytest` commands.
6.  **SageMaker Local Mode Testing:** Use SageMaker local mode for rapid testing of the images.

    *   Use the provided commands.
7.  **SageMaker Remote Tests:**
    *   Create an IAM role and add AmazonSageMakerFullAccess.
    *   Run the provided pytest commands.
8.  **SageMaker Benchmark Tests:**

    *   Create a `sm_benchmark_env_settings.config` file.
    *   Configure the environment, then run the `testrunner.py` script.

**Note:** SageMaker does not support tensorflow_inference py2 images.

```