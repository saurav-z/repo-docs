# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started quickly with pre-built, optimized Docker images for training and serving your machine learning models on AWS, and build custom images for a tailored approach.**

[Link to Original Repo:  https://github.com/aws/deep-learning-containers](https://github.com/aws/deep-learning-containers)

## Key Features

*   **Pre-built Images:**  Ready-to-use Docker images for popular deep learning frameworks like TensorFlow, PyTorch, and MXNet.
*   **Optimized Environments:**  Includes optimized libraries for CPU (Intel MKL) and GPU (Nvidia CUDA) instances, leading to faster training and inference.
*   **Seamless Integration:**  Designed for use with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Easy Deployment:**  Available in the Amazon Elastic Container Registry (Amazon ECR).
*   **Customization:**  Build custom images to meet your specific needs.

## Getting Started

This section guides you through building and testing Deep Learning Containers (DLCs), using an example of an MXNet GPU Python3 training container.

### Prerequisites

*   An active AWS account with appropriate permissions.  Recommended IAM roles are provided in the original README.
*   AWS CLI configured to access your account.
*   Docker client installed.
*   Access to an ECR repository (e.g., "beta-mxnet-training").

### Setup Steps

1.  **Clone the Repository:**
    ```shell
    git clone <repository_url>
    cd deep-learning-containers
    ```

2.  **Set Environment Variables:** Replace `<YOUR_ACCOUNT_ID>`, and `us-west-2` with your values.
    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```

3.  **Login to ECR:**
    ```shell
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
    ```

4.  **Create and Activate a Virtual Environment:**
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```

5.  **Perform Initial Setup:**
    ```shell
    bash src/setup.sh mxnet
    ```

## Building Your Image

### Understanding the Build Process

DLC image paths follow a consistent pattern, defined by `buildspec.yml` files within the framework directories. The `buildspec.yml` files specify the Dockerfile locations for different versions and configurations.

### Build Instructions

1.  **Build All Images (as defined in buildspec.yml):**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
    *   This command may take longer the first time as it downloads base layers and builds intermediate layers.  Subsequent runs are faster.

2.  **Build a Single Image:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
    *   Adjust the parameters for your specific requirements.

3.  **Available Parameters:**
    *   `--image_types`: `training` or `inference`
    *   `--device_types`: `cpu` or `gpu`
    *   `--py_versions`: `py2` or `py3`

    **Example: Build GPU training containers:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

### Upgrading the Framework Version

1.  **Modify `buildspec.yml`:**  Change the `version` key within the appropriate `buildspec.yml` file (e.g., `mxnet/training/buildspec.yml`) to the new framework version.

2.  **Dockerfile Path:**  Ensure the Dockerfile for the new version exists in the correct directory structure (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).

3.  **Build the container** as described in the building your image section.

### Adding Artifacts to Your Build Context

1.  **Define Context in `buildspec.yml`:**  Add the artifacts to your build context using the `context` key in your framework's `buildspec.yml` file.

2.  **Context Options:**
    *   `context`:  Makes the artifact available to all images.
    *   `training_context` or `inference_context`:  Makes the artifact available only for training or inference images, respectively.
    *   Image-specific `context`:  Makes the artifact available only for a specific image.

3.  **Build the container** as described in the building your image section.

### Adding a Package

Customize your containers by adding packages. See [Building AWS Deep Learning Containers Custom Images](custom_images.md) for more in-depth instructions.

1.  **Modify the Dockerfile:**  Add the `octopush` package installation command.

2.  **Build the container** as described in the building your image section.

## Running Tests Locally

This section covers how to run tests locally to validate changes before deployment, using pytest.

### Prerequisites

*   Ensure you have the images you want to test (likely pulled from ECR).
*   The testing requires a personal/team AWS account.
*   Install testing requirements.
    ```shell
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```

### Testing Instructions

1.  **Set Environment Variables:**
    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

2.  **Navigate to Test Directory:**
    ```shell
    cd test/dlc_tests
    ```

3.  **Run All Tests (in series):**
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
    *   Remove `-n=auto` to run tests sequentially.

4.  **Run a Specific Test File:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

5.  **Run a Specific Test Function:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

6.  **SageMaker Local Mode Tests:** Launch an EC2 instance and run these commands.

7.  **SageMaker Remote Tests:**  Configure IAM role, then run specific test commands, as described in the original README.

8.  **SageMaker Benchmark Tests:**  Create `sm_benchmark_env_settings.config` and run specific test commands, as described in the original README.