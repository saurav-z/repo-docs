# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Accelerate your machine learning projects with AWS Deep Learning Containers (DLCs), pre-built Docker images for training and serving models with popular frameworks.** ([Original Repo](https://github.com/aws/deep-learning-containers))

## Key Features

*   **Pre-built and Optimized:**  Ready-to-use Docker images pre-configured with popular deep learning frameworks like TensorFlow, PyTorch, and MXNet, along with optimized libraries for CPU and GPU instances.
*   **Framework Support:**  Comprehensive support for popular deep learning frameworks, including TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **GPU Acceleration:**  Leverage the power of NVIDIA GPUs with pre-configured CUDA libraries.
*   **CPU Optimization:**  Benefit from Intel MKL optimization for efficient CPU performance.
*   **Amazon Ecosystem Integration:** Designed for seamless use with Amazon SageMaker, EC2, ECS, and EKS.
*   **Easy Deployment:** Utilize Amazon Elastic Container Registry (ECR) for easy access and deployment.

## Getting Started

This guide outlines how to build and test AWS Deep Learning Containers (DLCs) on platforms such as Amazon SageMaker, EC2, ECS, and EKS.

### Prerequisites

1.  **AWS Account:** Ensure you have an active AWS account and have configured your environment for AWS CLI access.  We recommend using an IAM role for security. Required permissions are provided in the original README.
2.  **ECR Repository:** Create an ECR repository (e.g., "beta-mxnet-training" in `us-west-2`).
3.  **Docker:**  Install and configure the Docker client on your system.

### Setup

1.  **Clone the Repository:** Clone this repository to your local machine.
2.  **Set Environment Variables:** Set the following environment variables:

    ```bash
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
3.  **Login to ECR:**  Log in to your ECR repository using Docker:

    ```bash
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
4.  **Create and Activate Virtual Environment:** Create and activate a virtual environment and install the requirements:

    ```bash
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```

5.  **Perform Initial Setup:** Run the initial setup script:

    ```bash
    bash src/setup.sh mxnet
    ```

## Building Your Image

Follow these steps to build your custom Docker images using the provided build scripts.

1.  **Understand Buildspec:** Build specifications (e.g., `mxnet/training/buildspec.yml`) define the Dockerfile paths.
2.  **Build All Images:** Build all images specified in the buildspec:

    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

    Note: The first build will take longer as base layers are downloaded and intermediate layers are created.
3.  **Build a Single Image:** Build a specific image based on criteria:

    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
4.  **Image Type Options:**  Use the following flags to specify the image build options:

    ```bash
    --image_types <training/inference>
    --device_types <cpu/gpu>
    --py_versions <py2/py3>
    ```

### Upgrading the Framework Version

1.  **Modify `buildspec.yml`:** Update the `version` in your framework's buildspec file.
2.  **Update Dockerfile Path:** Ensure the corresponding Dockerfile exists at the correct path.
3.  **Rebuild the Container:** Build the container as described above.

### Adding Artifacts to Your Build Context

1.  **Add to `buildspec.yml`:** Add artifacts under `context`, `training_context`, or `inference_context` in your buildspec to include them in the build context.
2.  **Specify Source and Target:** Define the `source` (file path) and `target` (file name within the container) for each artifact.
3.  **Rebuild the Container:** Build the container as described above.

### Adding a Package

1.  **Modify Dockerfile:**  Add your package installation command (e.g., using `pip`) to the appropriate Dockerfile.
2.  **Rebuild the Container:** Build the container as described above.  See the provided example to see where to install.

## Running Tests Locally

Test your changes locally before deploying them to avoid resource usage and waiting.

1.  **Prepare your testing environment:**
    -Either clone the repo on an EC2 instance with the DLC images or locally.
    -Install the requirements for tests:

    ```shell script
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Set environment variables:**  Set DLC_IMAGES, and CODEBUILD_RESOLVED_SOURCE_VERSION as instructed in the original README.
3.  **Navigate to the tests directory:** `cd test/dlc_tests`
4.  **Run all tests (in series) for a specific platform:**

    ```bash
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto

    #EKS
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

7.  **SageMaker Local Mode Testing** as described in the original README.
8.  **SageMaker Remote Testing** as described in the original README.
9.  **SageMaker Benchmark Testing** as described in the original README.