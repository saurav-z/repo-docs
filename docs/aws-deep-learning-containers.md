# AWS Deep Learning Containers: Optimize Your Machine Learning Workflows

**Get pre-built, optimized Docker images for your deep learning tasks, streamlining development and deployment on AWS.**

[Link to Original Repo](https://github.com/aws/deep-learning-containers)

## Key Features

*   **Pre-built & Optimized:** Includes Docker images for popular deep learning frameworks like TensorFlow, PyTorch, and MXNet, pre-configured with necessary libraries (CUDA, MKL).
*   **AWS Integration:** Seamlessly integrates with Amazon SageMaker, EC2, ECS, and EKS for training, inference, and more.
*   **Flexible Deployment:** Available in Amazon Elastic Container Registry (ECR) for easy deployment across various AWS services.
*   **Framework Support:** Supports TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **GPU & CPU Optimization:**  Images optimized for both GPU (Nvidia CUDA) and CPU (Intel MKL) instances.

## Getting Started

This section details the steps to get started with AWS Deep Learning Containers, including building and testing.

### Prerequisites

1.  **AWS Account:** Ensure you have an active AWS account and have configured the AWS CLI with appropriate permissions.  Recommended IAM role with permissions:
    *   `AmazonEC2ContainerRegistryFullAccess`
    *   `AmazonEC2FullAccess`
    *   `AmazonEKSClusterPolicy`
    *   `AmazonEKSServicePolicy`
    *   `AmazonEKSServiceRolePolicy`
    *   `AWSServiceRoleForAmazonEKSNodegroup`
    *   `AmazonSageMakerFullAccess`
    *   `AmazonS3FullAccess`

2.  **Docker:** Ensure Docker is installed and running on your system.

### Setting Up Your Environment

1.  **Clone the Repository:** Clone the `deep-learning-containers` repository from GitHub.
2.  **Configure AWS CLI:** Configure your AWS CLI to access your account.
3.  **Create ECR Repository:** Create an ECR repository (e.g., "beta-mxnet-training") in your desired AWS region (e.g., us-west-2).
4.  **Set Environment Variables:**  Set the following environment variables, replacing the placeholders with your actual values:
    ```shell script
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
5.  **Login to ECR:** Authenticate with your ECR repository using the Docker CLI:
    ```shell script
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
6.  **Create Virtual Environment & Install Dependencies:**
    ```shell script
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
7.  **Initial Setup:**
    ```shell script
    bash src/setup.sh mxnet
    ```

## Building Your Image

The DLCs use a standardized structure for Dockerfiles. Build and customize your images using the following steps:

1.  **Locate the buildspec.yml:**  Build specifications are found in the `/mxnet/training/buildspec.yml` directory.
2.  **Build all images defined in buildspec.yml**
    ```shell script
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
3.  **Build a single image using the command:**
    ```shell script
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
4.  **Arguments for image building:**
    *   `--image_types <training/inference>`
    *   `--device_types <cpu/gpu>`
    *   `--py_versions <py2/py3>`

### Upgrading the Framework Version

To use a newer version of a framework:

1.  **Modify `buildspec.yml`:**  Update the `version` key in the corresponding `buildspec.yml` file (e.g., `mxnet/training/buildspec.yml`).
2.  **Ensure Dockerfile Exists:** Verify the Dockerfile for the new version is available in the appropriate directory (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  **Build the Container:** Build the updated container using the steps outlined in "Building Your Image."

### Adding Artifacts to Your Build Context

To include additional files in your build:

1.  **Add to buildspec.yml:** In the framework's `buildspec.yml` file, add the file under the `context` or `training_context` or `inference_context` key.  Specify the `source` (local file path) and `target` (path in the container) for the artifact.
2.  **Build the Container:** Rebuild the container using the steps outlined in "Building Your Image."

### Adding a Package

To add a package to your image:

1.  **Modify the Dockerfile:**  Add the `RUN pip install ...` command within the Dockerfile to install the desired package.
2.  **Build the Container:**  Rebuild the container.

## Running Tests Locally

Test your changes locally to save resources and speed up your development:

1.  **Install Test Requirements:**  Install testing dependencies.
    ```shell script
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Set Environment Variables:**
    ```shell script
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Change Directory:** Navigate to the test directory:
    ```shell script
    cd test/dlc_tests
    ```
4.  **Run All Tests:**
    *   **EC2:** `pytest -s -rA ec2/ -n=auto`
    *   **ECS:** `pytest -s -rA ecs/ -n=auto`
    *   **EKS:**
        ```shell script
        cd ../
        export TEST_TYPE=eks
        python test/testrunner.py
        ```
5.  **Run Specific Test File:**
    ```shell script
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```
6.  **Run Specific Test Function:**
    ```shell script
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

### SageMaker Local Mode Tests

1.  **Prerequisites:** Requires a local or EC2 instance with the latest Deep Learning AMI. Ensure images are pulled from ECR.
2.  **Run SageMaker Local Tests:**
    *   Navigate to the appropriate directory based on framework and job type in `test/sagemaker_tests/{framework}/{job_type}`.
    *   Install test requirements: `pip3 install -r requirements.txt`
    *   Run the appropriate test command: see original README for example pytest commands.

### SageMaker Remote Tests

1.  **Prerequisites:**
    *   Create an IAM role named "SageMakerRole" and attach the `AmazonSageMakerFullAccess` managed policy to it.
    *   Ensure you have the required permissions set up for the instance.
2.  **Run SageMaker Remote Tests:**
    *   Navigate to the appropriate directory based on framework and job type in `test/sagemaker_tests/{framework}/{job_type}`.
    *   Install test requirements: `pip3 install -r requirements.txt`
    *   Run the pytest command specified in the original README.

### SageMaker Benchmark Tests

1.  **Create Configuration File:** Create `sm_benchmark_env_settings.config` in the repository root.
2.  **Populate Configuration:** Add the needed environment variables, including  `DLC_IMAGES` (image URIs), `BUILD_CONTEXT`, `TEST_TYPE`, `CODEBUILD_RESOLVED_SOURCE_VERSION`, and `REGION`.
3.  **Source the Configuration:** `source sm_benchmark_env_settings.config`
4.  **Run Tests:** Run the testrunner or individual test files as described in the original README.