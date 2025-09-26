# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Accelerate your machine learning workflows with AWS Deep Learning Containers, pre-built and optimized Docker images for training and deploying models on Amazon SageMaker, EC2, ECS, and EKS.**

## Key Features

*   **Pre-built and Optimized:** Ready-to-use Docker images pre-configured with popular deep learning frameworks, NVIDIA CUDA, and Intel MKL.
*   **Framework Support:** Includes TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Platform Compatibility:** Works seamlessly with Amazon SageMaker, EC2, ECS, and EKS.
*   **GPU and CPU Support:** Optimized for both GPU (NVIDIA) and CPU (Intel) instances.
*   **Easy Deployment:** Available in Amazon Elastic Container Registry (Amazon ECR) for simplified deployment.

For a complete list of available images, see the [Available Deep Learning Containers Images](available_images.md) in this repository.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the framework version](#upgrading-the-framework-version)
*   [Adding artifacts to your build context](#adding-artifacts-to-your-build-context)
*   [Adding a package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

### Getting Started

This section outlines the steps to build and test AWS Deep Learning Containers on various platforms, including Amazon SageMaker, EC2, ECS, and EKS.

**Example: Building an MXNet GPU Python3 Training Container**

1.  **Prerequisites:**
    *   An AWS account and configured AWS CLI access (IAM role recommended).  Required IAM Permissions are listed in the original README.
    *   An ECR repository (e.g., "beta-mxnet-training") in your desired region (e.g., us-west-2).
    *   Docker client installed and set up on your system.

2.  **Setup:**

    *   Clone the repository.
    *   Set the necessary environment variables:
        ```bash
        export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
        export REGION=us-west-2
        export REPOSITORY_NAME=beta-mxnet-training
        ```
    *   Log in to ECR:
        ```bash
        aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
        ```
    *   Create a virtual environment and install dependencies:
        ```bash
        python3 -m venv dlc
        source dlc/bin/activate
        pip install -r src/requirements.txt
        ```
    *   Perform initial setup:
        ```bash
        bash src/setup.sh mxnet
        ```

### Building your image

The paths to the dockerfiles follow a specific pattern e.g., mxnet/training/docker/\<version>/\<python_version>/Dockerfile.\<processor>

These paths are specified by the buildspec.yml residing in mxnet/training/buildspec.yml i.e. \<framework>/<training|inference>/buildspec.yml.
If you want to build the dockerfile for a particular version, or introduce a new version of the framework, re-create the
folder structure as per above and modify the buildspec.yml file to specify the version of the dockerfile you want to build.

1.  **Build all images:**

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
3.  **Arguments:**
    *   `--image_types`: `training/inference`
    *   `--device_types`: `cpu/gpu`
    *   `--py_versions`: `py2/py3`
4.  **Example for all GPU training containers:**

    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```
### Upgrading the framework version
1.  **Update `buildspec.yml`**: Change the `version` key in the buildspec.yml file (e.g., `mxnet/training/buildspec.yml`) to the new framework version (e.g., 1.7.0).
2.  **Docker File Path**: Ensure the Dockerfile for the new version exists in the correct path (e.g.,  `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  **Build the container**: Build the container following the steps described above.
### Adding artifacts to your build context
1.  **Add artifact to context**:
     Add the artifact in the framework buildspec file under the context key.
    ```yaml
    # mxnet/training/buildspec.yml
     19 context:
     20   README.xyz: *<---- Object name (Can be anything)*
     21     source: README-context.rst *<--- Path for the file to be copied*
     22     target: README.rst *<--- Name for the object in** the build context*
    ```
2.  **Define context**:
     To restrict artifacts to all images or training/inference images, use the context key.
    ```yaml
     19   context:
        .................
     23       training_context: &TRAINING_CONTEXT
     24         README.xyz:
     25           source: README-context.rst
     26           target: README.rst
        ...............
    ```
3.  **Single container context**:
     To add context to a single container use the context key.
    ```yaml
     41   images:
     42     BuildMXNetCPUTrainPy3DockerImage:
     43       <<: *TRAINING_REPOSITORY
              .......................
     50       context:
     51         <<: *TRAINING_CONTEXT
     52         README.xyz:
     53           source: README-context.rst
     54           target: README.rst
    ```
4.  **Build the container**: Build the container following the steps described above.
### Adding a package
1.  **Add a package to the Dockerfile**:
    Update the Dockerfile (e.g. `mxnet/training/docker/1.6.0/py3/Dockerfile.gpu`) to include the package.
    ```dockerfile
    139 RUN ${PIP} install --no-cache --upgrade \
    140     keras-mxnet==2.2.4.2 \
    ...........................
    160     awscli \
    161     octopush
    ```
2.  **Build the container**: Build the container following the steps described above.

### Running Tests Locally

This section describes how to run tests locally to validate changes before deployment, using pytest.

1.  **Prerequisites:**
    *   Images to test (pull from ECR if necessary).
    *   Test requirements installed:
        ```bash
        cd deep-learning-containers/
        pip install -r src/requirements.txt
        pip install -r test/requirements.txt
        ```
2.  **Environment variables:**
    *   `DLC_IMAGES`: Space-separated list of ECR URIs for images to test.
    *   `CODEBUILD_RESOLVED_SOURCE_VERSION`: Unique identifier for test resources.
    *   `PYTHONPATH`: Absolute path to the `src/` folder.

    Example:
    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Change directory:**
    ```bash
    cd test/dlc_tests
    ```
4.  **Run all tests (in series):**

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

    (Remove `-n=auto` for sequential execution.)
5.  **Run a specific test file:**

    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```
6.  **Run a specific test function:**

    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```
7.  **Run SageMaker local mode tests**:
    *   Run the commands and configuration listed in the original README under 'Running tests locally'
8.  **Run SageMaker remote tests**:
    *   Run the commands and configuration listed in the original README under 'Running tests locally'
9.  **Run SageMaker benchmark tests**:
    *   Run the commands and configuration listed in the original README under 'Running tests locally'

For more detailed information, refer to the [AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers) repository.