# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started with pre-built, optimized Docker images for training and deploying your machine learning models with AWS Deep Learning Containers (DLCs).**  [Original Repo](https://github.com/aws/deep-learning-containers)

## Key Features:

*   **Pre-built and Optimized:**  Benefit from pre-configured environments with popular deep learning frameworks like TensorFlow, PyTorch, and MXNet, along with optimized libraries like Nvidia CUDA (for GPU instances) and Intel MKL (for CPU instances).
*   **Framework Support:** Supports popular machine learning frameworks, including TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Amazon Integration:** Seamlessly integrate with Amazon SageMaker for training, inference, and transformations, as well as compatibility with Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Available in ECR:** DLCs are readily available in the Amazon Elastic Container Registry (Amazon ECR) for easy access and deployment.
*   **Flexible Deployment:** Designed for use across various AWS services, including SageMaker, EC2, ECS, and EKS.

## Table of Contents

1.  [Getting Started](#getting-started)
2.  [Building Your Image](#building-your-image)
3.  [Upgrading Framework Version](#upgrading-the-framework-version)
4.  [Adding Artifacts to Your Build Context](#adding-artifacts-to-your-build-context)
5.  [Adding a Package](#adding-a-package)
6.  [Running Tests Locally](#running-tests-locally)

### Getting Started

This section outlines the setup for building and testing DLCs on Amazon SageMaker, EC2, ECS, and EKS.

1.  **Prerequisites:**
    *   An AWS account with the necessary permissions.
    *   The following IAM managed permissions are recommended:
        *   AmazonEC2ContainerRegistryFullAccess
        *   AmazonEC2FullAccess
        *   AmazonEKSClusterPolicy
        *   AmazonEKSServicePolicy
        *   AmazonEKSServiceRolePolicy
        *   AWSServiceRoleForAmazonEKSNodegroup
        *   AmazonSageMakerFullAccess
        *   AmazonS3FullAccess
    *   Docker client set up on your system.
    *   AWS CLI configured to access your AWS account.
2.  **Setup:**
    *   Create an ECR repository (e.g., "beta-mxnet-training" in us-west-2).
    *   Clone the repository and set environment variables.
    *   Log in to ECR.
    *   Create a virtual environment and install requirements.
    *   Perform initial setup using the `src/setup.sh` script.

### Building Your Image

Instructions for building Docker images based on `buildspec.yml` configuration files.

1.  **Building All Images:**
    *   Use the command: `python src/main.py --buildspec <buildspec_path> --framework <framework_name>`
2.  **Building a Single Image:**
    *   Use the command: `python src/main.py --buildspec <buildspec_path> --framework <framework_name> --image_types <training/inference> --device_types <cpu/gpu> --py_versions <py2/py3>`
3.  **Buildspec.yml:**  Modify the buildspec.yml to build a Dockerfile for a specific version of the framework.

### Upgrading Framework Version

Steps to upgrade the framework version in your DLC images.

1.  **Modify `buildspec.yml`:**  Update the `version` key within the `buildspec.yml` file for the desired framework.
2.  **Dockerfile Location:** Ensure the Dockerfile for the new version exists in the correct directory structure.
3.  **Build Container:**  Build the container using the build commands described above.

### Adding Artifacts to Your Build Context

How to include artifacts in the build context.

1.  **Add to `buildspec.yml`:** Define the artifact in the `context`, `training_context`, or `inference_context` section of the `buildspec.yml` file, specifying the source and target paths.
2.  **Build Container:**  Build the container using the build commands described above.

### Adding a Package

Instructions on adding packages to your DLC images.

1.  **Modify Dockerfile:**  Add the package installation command (e.g., `RUN pip install ...`) to the Dockerfile.
2.  **Build Container:**  Build the container using the build commands described above.

### Running Tests Locally

How to run local tests to validate your images.

1.  **Prerequisites:**
    *   Access to a personal/team AWS account.
    *   Images you want to test, pulled from ECR.
    *   Install requirements ( `pip install -r src/requirements.txt` and  `pip install -r test/requirements.txt`)
2.  **Configure Environment Variables:**
    *   Set `DLC_IMAGES` to a space-separated list of ECR URIs.
    *   Set `PYTHONPATH` to the absolute path of the `src/` folder.
    *   Set `CODEBUILD_RESOLVED_SOURCE_VERSION` to a unique identifier.
3.  **Run Tests:**
    *   Navigate to the `test/dlc_tests` directory.
    *   Run tests using `pytest` commands:

        *   To run all tests (in series) associated with your image for a given platform, use the following command:
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
        *   To run a specific test file: `pytest -s <test_file_path>`
        *   To run a specific test function: `pytest -s <test_file_path>::<test_function_name>`
    *   For SageMaker Local Mode tests, follow the detailed steps provided in the original README.
    *   For SageMaker remote tests, follow the detailed steps provided in the original README.
    *   For SageMaker benchmark tests, follow the detailed steps provided in the original README.