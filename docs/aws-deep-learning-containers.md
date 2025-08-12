# AWS Deep Learning Containers: Pre-built Docker Images for Machine Learning

**Quickly and easily train and deploy your machine learning models with optimized Docker images from AWS, eliminating the complexities of environment setup.**  [View the original repository](https://github.com/aws/deep-learning-containers)

## Key Features:

*   **Pre-built and Optimized:** Includes TensorFlow, TensorFlow 2, PyTorch, and MXNet, along with optimized libraries (CUDA, Intel MKL).
*   **Seamless Integration:** Designed for use with Amazon SageMaker, EC2, ECS, and EKS.
*   **Easy Deployment:** Available in Amazon Elastic Container Registry (Amazon ECR).
*   **GPU and CPU Support:** Images tailored for both GPU-enabled and CPU-based instances.
*   **Comprehensive Testing:** Robust testing framework using pytest for EC2, ECS, EKS, and SageMaker local/remote modes.

## Getting Started

This section guides you through setting up the necessary environment to build and test the DLCs on platforms like Amazon SageMaker, EC2, ECS, and EKS. We'll use the example of building an MXNet GPU Python3 training container.

**Prerequisites:**

*   An active AWS account with appropriate permissions. Configure your AWS CLI with an IAM role for secure access. Recommended permissions include:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
*   Docker installed and configured on your system.

**Setup Steps:**

1.  **Set up your environment:**
    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
2.  **Create an ECR repository:** Create an ECR repository (e.g., "beta-mxnet-training" in us-west-2).
3.  **Authenticate with ECR:**
    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
4.  **Clone the repository:**  Make sure the code is in the current directory
5.  **Create a virtual environment and install dependencies:**
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
6.  **Perform initial setup:**
    ```shell
    bash src/setup.sh mxnet
    ```

## Building Your Image

The DLCs use a consistent structure for Dockerfiles (e.g., `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`). The `buildspec.yml` file within each framework's directory (`<framework>/<training|inference>/buildspec.yml`) specifies the Dockerfile paths.

**Building Docker Images:**

1.  **Build all images from buildspec.yml:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  **Build a specific image:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```

**Image Arguments:**

*   `--image_types`:  `training` or `inference`
*   `--device_types`: `cpu` or `gpu`
*   `--py_versions`: `py2` or `py3`

## Upgrading the Framework Version

To update the framework version (e.g., MXNet from 1.6.0 to 1.7.0), modify the `buildspec.yml` file and ensure the corresponding Dockerfile exists.

1.  **Edit `buildspec.yml`:**
    ```yaml
    # mxnet/training/buildspec.yml
    # ...
    version: &VERSION 1.7.0  # Change to the new version
    ```
2.  **Verify Dockerfile:** Confirm the Dockerfile exists (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  **Build the container:** Build the container using the steps in the Building Your Image section.

## Adding Artifacts to Your Build Context

To include files in your build context:

1.  **Add Artifacts in `buildspec.yml`:** Add the artifact and specify its source and target paths:
    ```yaml
    # mxnet/training/buildspec.yml
    context:
      README.xyz:
        source: README-context.rst
        target: README.rst
    ```
2.  **Choose the Context:**  Add artifacts under `context`, `training_context`, or `inference_context` based on your needs.
3.  **Build the container:** Build the container using the steps in the Building Your Image section.

## Adding a Package

To add packages to your image, modify the Dockerfile:

1.  **Modify the Dockerfile:**  Add the package installation command to the Dockerfile (e.g., using `pip install`).
    ```dockerfile
    RUN ${PIP} install --no-cache --upgrade \
        keras-mxnet==2.2.4.2 \
        awscli \
        octopush # Add the new package
    ```
2.  **Build the container:** Build the container using the steps in the Building Your Image section.

## Running Tests Locally

To test your changes locally, use the pytest framework.

**Prerequisites:**
*   Images you want to test locally (pulled from ECR).
*   Test requirements installed.

**Testing Steps:**

1.  **Set up your environment:**

    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
2.  **Navigate to test directory:**
    ```shell
    cd test/dlc_tests
    ```
3.  **Run all tests (in series) for a given platform:**
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
4.  **Run a specific test file:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```
5.  **Run a specific test function:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

6.  **SageMaker local mode tests:**  Requires an EC2 instance with the latest Deep Learning AMI.
    *   Clone the repository and checkout the desired branch.
    *   Login into ECR.
    *   Navigate to the appropriate directory (`test/sagemaker_tests/{framework}/{job_type}`).
    *   Run the tests using pytest commands like the examples below.
    * For example:
        ```shell
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
        --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
        --py-version 3
        ```
7.  **SageMaker remote tests:**
    *   Create an IAM role "SageMakerRole" with AmazonSageMakerFullAccess.
    *   Navigate to the appropriate directory.
    *   Run tests using pytest commands.
     * For example:
       ```shell
       pytest integration/sagemaker/test_mnist.py \
       --region us-west-2 --docker-base-name mxnet-training \
       --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
       --instance-type ml.g5.12xlarge
       ```
8.  **SageMaker benchmark tests:**
    *   Create `sm_benchmark_env_settings.config`.
    *   Populate with environment variables.
    *   Run  `source sm_benchmark_env_settings.config`.
    *   Execute the test runner.

**Note:**  SageMaker does not support tensorflow_inference py2 images.