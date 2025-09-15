# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started with pre-built, optimized Docker images for deep learning frameworks like TensorFlow, PyTorch, and MXNet, simplifying your machine learning workflows on AWS!**  [See the original repo here](https://github.com/aws/deep-learning-containers).

## Key Features:

*   **Pre-built and Optimized:** Benefit from pre-configured environments with popular deep learning frameworks, NVIDIA CUDA (for GPU instances), and Intel MKL (for CPU instances).
*   **Framework Support:**  Includes images for TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **AWS Integration:** Designed for seamless use with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Amazon ECR Availability:** Easily access and deploy containers from Amazon Elastic Container Registry (ECR).
*   **Simplified Development:**  Provides readily available images, reducing the time and effort needed to set up your machine learning environment.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the Framework Version](#upgrading-the-framework-version)
*   [Adding Artifacts to Your Build Context](#adding-artifacts-to-your-build-context)
*   [Adding a Package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

### Getting Started

This section outlines the steps required to build and test the Deep Learning Containers on various platforms, including Amazon SageMaker, EC2, ECS, and EKS.  We'll use an example of building an ***MXNet GPU python3 training*** container.

**Prerequisites:**

*   An active AWS account with the necessary permissions.  Recommended permissions include:

    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
*   Docker client installed and configured on your system.

**Steps:**

1.  **Configure AWS CLI:** Set up your AWS CLI to access your account, preferably using an IAM role.
2.  **Create ECR Repository:** Create an ECR repository named `beta-mxnet-training` in the `us-west-2` region.
3.  **Clone Repository & Set Environment Variables:**

    ```shell
    git clone https://github.com/aws/deep-learning-containers.git
    cd deep-learning-containers
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
4.  **Login to ECR:**

    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
5.  **Create and Activate Virtual Environment & Install Dependencies:**

    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
6.  **Perform Initial Setup:**

    ```shell
    bash src/setup.sh mxnet
    ```

### Building Your Image

The paths to the Dockerfiles follow a structured pattern based on framework, training/inference, version, Python version, and processor type (e.g., `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`).  The buildspec.yml files (e.g.,  `mxnet/training/buildspec.yml`) define these paths.

1.  **Build All Images:** To build all Dockerfiles specified in the `buildspec.yml`, use:

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  **Build a Single Image:**  To build a specific image, use:

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```

3.  **Image Type, Device Type, and Python Version Arguments:**

    *   `--image_types`:  `training` or `inference`
    *   `--device_types`:  `cpu` or `gpu`
    *   `--py_versions`: `py2` or `py3`

4.  **Example: Build all GPU training containers:**

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

### Upgrading the Framework Version

When a new framework version is released (e.g., MXNet 1.7.0), you need to update the `buildspec.yml` file:

1.  **Modify `buildspec.yml`:** Locate the `version` key in your `buildspec.yml` file and change the version number (e.g., from `1.6.0` to `1.7.0`).
2.  **Ensure Dockerfile Exists:**  Verify that the corresponding Dockerfile exists (e.g.,  `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  **Rebuild the Container:** Build the container using the instructions above.

### Adding Artifacts to Your Build Context

To include files (e.g.,  `README-context.rst`) in the build context for your images:

1.  **Add to `context` Key in `buildspec.yml`:**  Add the artifact under the `context` key in your `buildspec.yml` file.

    ```yaml
    context:
      README.xyz:  # Object name
        source: README-context.rst  # Source file
        target: README.rst  # Destination in build context
    ```

2.  **Context Scope:**  You can add to `training_context` or `inference_context` for images of that type or to a specific image via its context key.

### Adding a Package

To add a package to your container (e.g., `octopush` to the MXNet 1.6.0 py3 GPU image):

1.  **Modify the Dockerfile:**  Add the `octopush` package to the `RUN ${PIP} install` command in the relevant Dockerfile.

    ```dockerfile
    RUN ${PIP} install --no-cache --upgrade \
        ...
        awscli \
        octopush
    ```
2.  **Rebuild the Container:** Build the container with the modified Dockerfile.

### Running Tests Locally

You can run tests locally to iterate quickly and avoid using extraneous resources.

1.  **Prerequisites:**
    *   Make sure you have the images that you want to test locally (likely need to pull them from ECR).
    *   Install test requirements
        ```shell
        cd deep-learning-containers/
        pip install -r src/requirements.txt
        pip install -r test/requirements.txt
        ```
    *   Ensure you have your AWS account setup.
2.  **Set Environment Variables:**

    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Navigate to Test Directory:**

    ```shell
    cd test/dlc_tests
    ```
4.  **Run All Tests (in series) for a platform:**

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
5.  **Run a Specific Test File:**

    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

6.  **Run a Specific Test Function:**

    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

7.  **SageMaker Local Mode Tests:**

    *   Launch an EC2 instance with a Deep Learning AMI.
    *   Clone the repository and checkout the desired branch
    *   Login into ECR repo.
    *   Navigate to  the appropriate directory.
        ```shell
        cd test/sagemaker_tests/mxnet/training/
        pip3 install -r requirements.txt
        ```
    *   Run SageMaker local integration tests
        ```shell
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
         --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
         --py-version 3
        ```
8.  **SageMaker Remote Tests:**

    *   Create an IAM role named “SageMakerRole” with AmazonSageMakerFullAccess permissions.
    *   Navigate to  the appropriate directory
        ```shell
        cd test/sagemaker_tests/mxnet/training/
        pip3 install -r requirements.txt
        ```
    *   Run the appropriate pytest command.

9.  **SageMaker Benchmark Tests:**
    *   Create `sm_benchmark_env_settings.config` and set your image URI, test type, and other settings.
    *   Source this config.
    *   Run tests using `python test/testrunner.py` or run pytest commands inside `test/dlc_tests`.