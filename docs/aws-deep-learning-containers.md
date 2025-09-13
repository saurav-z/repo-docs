# AWS Deep Learning Containers

**Get started quickly with pre-built, optimized Docker images for machine learning, powered by AWS Deep Learning Containers (DLCs).** ([Back to Original Repo](https://github.com/aws/deep-learning-containers))

AWS Deep Learning Containers (DLCs) provide pre-packaged Docker images for training and serving machine learning models using popular frameworks. These images are optimized for performance on AWS infrastructure, including GPU instances, and are readily available in the Amazon Elastic Container Registry (Amazon ECR).

## Key Features:

*   **Pre-built and Optimized:** DLCs offer optimized environments with popular machine learning frameworks like TensorFlow, TensorFlow 2, PyTorch, and MXNet, along with NVIDIA CUDA (for GPU instances) and Intel MKL (for CPU instances).
*   **Integration with AWS Services:** Seamlessly integrates with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS, simplifying deployment and scaling.
*   **Wide Availability:**  Available in Amazon ECR for easy access and use.
*   **Flexible Use Cases:**  Supports training, inference, and transformations.

## Getting Started

This section provides instructions to build and test DLCs on various platforms, including Amazon SageMaker, EC2, ECS, and EKS.

### Prerequisites:

*   An active AWS account with appropriate permissions.  Recommended IAM role configuration includes permissions for ECR, EC2, EKS, SageMaker, and S3 (see original README for specific managed policy details).
*   Docker installed and configured on your system.

### Example: Building a MXNet GPU Python3 Training Container

1.  **Set up your environment:**
    *   Clone the repository.
    *   Set the following environment variables:

        ```bash
        export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
        export REGION=us-west-2
        export REPOSITORY_NAME=beta-mxnet-training
        ```

2.  **Authenticate with ECR:**
    ```bash
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```

3.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```

4.  **Perform initial setup:**
    ```bash
    bash src/setup.sh mxnet
    ```

## Building Your Image

The build process uses `buildspec.yml` files to define the Dockerfile paths and build configurations.

1.  **Build all images defined in buildspec.yml:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  **Build a single image:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
    *   Use `--image_types`, `--device_types`, and `--py_versions` arguments to customize the build.  Possible values for these arguments are specified in the original README.

## Upgrading the Framework Version

1.  Modify the `version` in the appropriate `buildspec.yml` file (e.g., `mxnet/training/buildspec.yml`).
2.  Ensure the corresponding Dockerfile exists in the correct directory structure. (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`)
3.  Build the container as described above.

## Adding Artifacts and Packages

Instructions on how to add artifacts to your build context (using the `context` key in buildspec files) and how to add packages to your image are provided in the original README.

## Running Tests Locally

The project uses pytest for testing.

1.  **Install test requirements:**
    ```bash
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Set environment variables:**
    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

3.  **Navigate to the test directory:**
    ```bash
    cd test/dlc_tests
    ```

4.  **Run tests:**  Examples are provided in the original README for running tests on EC2, ECS, and EKS, including how to run specific test files and functions.
5. **SageMaker local mode tests:** instructions for running tests in this mode are in the original README.
6.  **SageMaker remote tests:**  Instructions are provided for running SageMaker remote integration tests and SageMaker benchmark tests, including pre-requisites such as creating a SageMakerRole and example commands to run the tests.