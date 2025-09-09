# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started quickly with pre-built, optimized Docker images for training and serving machine learning models with AWS Deep Learning Containers (DLCs).** [Visit the original repo](https://github.com/aws/deep-learning-containers) for the latest updates.

## Key Features

*   **Framework Support:** TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Optimized Environments:** Pre-configured with CUDA (for GPU) and Intel MKL (for CPU) libraries for optimal performance.
*   **Deployment Flexibility:** Compatible with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Easy Access:** Available in the Amazon Elastic Container Registry (Amazon ECR).

## Sections

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the framework version](#upgrading-the-framework-version)
*   [Adding artifacts to your build context](#adding-artifacts-to-your-build-context)
*   [Adding a package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

### Getting Started

This section provides instructions for setting up and testing the DLCs on various platforms, including Amazon SageMaker, EC2, ECS, and EKS.

1.  **Prerequisites:** Ensure you have an AWS account and the necessary permissions configured using the AWS CLI. We recommend using an IAM role. For testing purposes, the following managed policies should suffice: `AmazonEC2ContainerRegistryFullAccess`, `AmazonEC2FullAccess`, `AmazonEKSClusterPolicy`, `AmazonEKSServicePolicy`, `AmazonEKSServiceRolePolicy`, `AWSServiceRoleForAmazonEKSNodegroup`, `AmazonSageMakerFullAccess`, and `AmazonS3FullAccess`.
2.  **Create ECR Repository:** Create an ECR repository (e.g., "beta-mxnet-training") in your desired AWS region.
3.  **Docker Setup:** Ensure you have Docker installed and configured on your system.
4.  **Clone and Configure:** Clone the repository and set the following environment variables:
    ```bash
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
5.  **ECR Login:** Authenticate with ECR. Replace `<REGION>` with your desired AWS region:
    ```bash
    aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.<REGION>.amazonaws.com
    ```
6.  **Virtual Environment and Installation:** Create a virtual environment and install the required packages:
    ```bash
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
7.  **Initial Setup:** Execute the setup script.
    ```bash
    bash src/setup.sh mxnet
    ```

### Building Your Image

Build and customize your Docker images using the provided build specifications. The build process uses `buildspec.yml` files located in each framework's directory (e.g., `mxnet/training/buildspec.yml`).

1.  **Build All Images:** Build all images specified in the `buildspec.yml` file:
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  **Build a Specific Image:** Build a specific image by specifying image type, device type, and Python version:
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
3.  **Image Type, Device Type, and Python version**: possible values are as follows:
    ```
    --image_types <training/inference>
    --device_types <cpu/gpu>
    --py_versions <py2/py3>
    ```

### Upgrading the framework version

To upgrade to a new framework version (e.g., MXNet 1.7.0):

1.  **Modify `buildspec.yml`:** Update the `version` key in the relevant `buildspec.yml` file.
2.  **Ensure Dockerfile:** Verify the existence of the corresponding Dockerfile (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  **Build the Container:** Build the updated container using the instructions above.

### Adding artifacts to your build context

Add extra artifacts to your build context by modifying the `buildspec.yml` file.

1.  **Add to Context:** Add your artifact under the `context` key:
    ```yaml
    context:
      README.xyz:
        source: README-context.rst
        target: README.rst
    ```
2.  **Training or Inference Context:** You can also specify the artifact to only the training or inference image.
3.  **Single Container Context:** Add to the image context
4.  **Build the Container:** Build the updated container using the instructions above.

### Adding a package

Customize your images by adding packages to the Dockerfile.

1.  **Modify Dockerfile:** Add a package installation command to the Dockerfile (e.g., using `pip install`).
2.  **Build the Container:** Build the updated container using the instructions above.

### Running Tests Locally

Test your changes locally before deploying them.

1.  **Prerequisites:** Ensure you have the required images and install testing dependencies.
    ```bash
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Set Environment Variables:** Export `DLC_IMAGES`, `PYTHONPATH`, and `CODEBUILD_RESOLVED_SOURCE_VERSION`:
    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Navigate to Test Directory:** Change your directory to `test/dlc_tests`.
4.  **Run Tests (EC2, ECS):** Run all tests for a given platform.
    ```bash
    pytest -s -rA ec2/ -n=auto
    pytest -s -rA ecs/ -n=auto
    ```
    Remove `-n=auto` to run the tests sequentially.
5.  **Run Tests (EKS):** To run EKS tests, set `TEST_TYPE`.
    ```bash
    cd ../
    export TEST_TYPE=eks
    python test/testrunner.py
    ```
6.  **Run Specific Test File:**
    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```
7.  **Run Specific Test Function:**
    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```
8.  **Run SageMaker Local Mode Tests:**
    *   Clone the GitHub branch on an EC2 instance with the latest Deep Learning AMI.
    *   Login to the ECR repository.
    *   Change to the appropriate directory based on the framework and job type.
    *   Use the `pytest` command.
9.  **Run SageMaker Remote Tests:**
    *   Create an IAM role named "SageMakerRole."
    *   Change to the appropriate directory.
    *   Use the appropriate pytest command.
10. **Run SageMaker Benchmark Tests:**
    *   Create `sm_benchmark_env_settings.config`
    *   Add the necessary configurations.
    *   Run the test.