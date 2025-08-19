# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Accelerate your machine learning projects with AWS Deep Learning Containers (DLCs), pre-built Docker images optimized for training and deploying models.**  ([Original Repo](https://github.com/aws/deep-learning-containers))

**Key Features:**

*   **Pre-built and Optimized:** Ready-to-use Docker images for popular deep learning frameworks.
*   **Framework Support:**  Includes TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Hardware Acceleration:** Optimized for both CPU (with Intel MKL) and GPU (with Nvidia CUDA) instances.
*   **Amazon Integration:** Designed for seamless use with Amazon SageMaker, EC2, ECS, and EKS.
*   **Easy Deployment:**  Available in Amazon Elastic Container Registry (ECR) for quick deployment.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the Framework Version](#upgrading-the-framework-version)
*   [Adding Artifacts to Your Build Context](#adding-artifacts-to-your-build-context)
*   [Adding a Package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

## Getting Started

This section outlines the setup for building and testing DLCs on Amazon SageMaker, EC2, ECS, and EKS.  This example focuses on building an MXNet GPU python3 training container.

**Prerequisites:**

*   An AWS account with configured CLI access (IAM role recommended, with the permissions detailed in the original README).
*   An ECR repository (e.g., "beta-mxnet-training" in us-west-2).
*   Docker installed on your system.

**Steps:**

1.  **Clone the repository** and set the following environment variables:

    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```

2.  **Login to ECR:**

    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```

3.  **Create a virtual environment** and install requirements:

    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```

4.  **Perform initial setup:**

    ```shell
    bash src/setup.sh mxnet
    ```

## Building Your Image

The DLC build process uses `buildspec.yml` files to define Dockerfile paths. These paths follow a consistent pattern.

1.  **Build all images:**

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

    (First run will take longer as it downloads base layers).
2.  **Build a single image:**

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```

3.  **Specify Image Types, Device Types, and Python Versions:** Use the `--image_types`, `--device_types`, and `--py_versions` arguments (comma-separated lists).

    *   `--image_types`:  `training`, `inference`
    *   `--device_types`: `cpu`, `gpu`
    *   `--py_versions`: `py2`, `py3`

4.  **Example:** Build all GPU training containers with Python 3:

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

## Upgrading the Framework Version

To update to a new framework version (e.g., MXNet 1.7.0):

1.  **Modify `buildspec.yml`:** Change the `version` key in the appropriate `buildspec.yml` file.
2.  **Update Dockerfile path (if needed):** Ensure the Dockerfile for the new version exists at the expected path.
3.  **Build the container** as described above.

## Adding Artifacts to Your Build Context

To include files in the build context (e.g., a README file):

1.  **Add to `buildspec.yml`:** Define the artifact under the `context`, `training_context`, or `inference_context` key in the  `buildspec.yml` file, specifying the source and target paths.
2.  **Build the container** as described above.

## Adding a Package

To add a package to your image:

1.  **Modify the Dockerfile:** Add the package installation command (e.g., `pip install`) to the Dockerfile.
2.  **Build the container** as described above.

## Running Tests Locally

Run tests locally to validate your changes without using too many resources.

1.  **Prerequisites:**
    *   Ensure the images to be tested are available locally (pull from ECR).
    *   Install the requirements for tests:

        ```shell
        cd deep-learning-containers/
        pip install -r src/requirements.txt
        pip install -r test/requirements.txt
        ```

2.  **Set environment variables:**

    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

3.  **Navigate to the test directory:**

    ```shell
    cd test/dlc_tests
    ```

4.  **Run tests (series) for a specific platform:**

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

    (Remove `-n=auto` for sequential tests)
5.  **Run a specific test file:**

    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

6.  **Run a specific test function:**

    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```
7.  **Run SageMaker local mode tests**
8.  **Run SageMaker remote tests**
9.  **Run SageMaker benchmark tests**