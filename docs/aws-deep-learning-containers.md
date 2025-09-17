## AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Accelerate your machine learning workflows with AWS Deep Learning Containers (DLCs), pre-built and optimized Docker images for training and deploying models.**  [View the original repository](https://github.com/aws/deep-learning-containers).

**Key Features:**

*   **Pre-built and Optimized:** Benefit from pre-configured environments with popular deep learning frameworks and libraries.
*   **Framework Support:** Includes TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Hardware Acceleration:**  Optimized for both CPU and GPU instances with Nvidia CUDA and Intel MKL.
*   **Amazon ECR Availability:** Easily access and deploy DLCs from Amazon Elastic Container Registry (ECR).
*   **SageMaker Integration:** Seamlessly integrates with Amazon SageMaker for training, inference, and more.
*   **Platform Compatibility:** Tested on Amazon EC2, Amazon ECS, and Amazon EKS.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the Framework Version](#upgrading-the-framework-version)
*   [Adding Artifacts to Your Build Context](#adding-artifacts-to-your-build-context)
*   [Adding a Package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

### Getting Started

This section provides instructions to build and test the DLCs on various platforms like Amazon SageMaker, EC2, ECS, and EKS. This example focuses on building a MXNet GPU python3 training container.

*   **Prerequisites:**

    *   An AWS account with appropriate permissions (IAM role recommended).  The following IAM managed permissions should suffice:
        *   [AmazonEC2ContainerRegistryFullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess)
        *   [AmazonEC2FullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEC2FullAccess)
        *   [AmazonEKSClusterPolicy](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEKSClusterPolicy)
        *   [AmazonEKSServicePolicy](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEKSServicePolicy)
        *   [AmazonEKSServiceRolePolicy](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEKSServiceRolePolicy)
        *   [AWSServiceRoleForAmazonEKSNodegroup](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AWSServiceRoleForAmazonEKSNodegroup)
        *   [AmazonSageMakerFullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonSageMakerFullAccess)
        *   [AmazonS3FullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonS3FullAccess)
    *   An ECR repository (e.g., “beta-mxnet-training” in us-west-2).
    *   Docker installed and configured.

*   **Steps:**

    1.  Clone the repository and set environment variables:
        ```shell
        export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
        export REGION=us-west-2
        export REPOSITORY_NAME=beta-mxnet-training
        ```
    2.  Log in to ECR:
        ```shell
        aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
        ```
    3.  Create and activate a virtual environment, and install requirements:
        ```shell
        python3 -m venv dlc
        source dlc/bin/activate
        pip install -r src/requirements.txt
        ```
    4.  Perform initial setup:
        ```shell
        bash src/setup.sh mxnet
        ```

### Building Your Image

DLC images are built from Dockerfiles using a specific file structure.  `buildspec.yml` files define the build process.

1.  To build all images defined in a `buildspec.yml` locally:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  To build a single image, use the following command with specific parameters:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
    *   Available arguments:

        *   `--image_types`: `<training/inference>`
        *   `--device_types`: `<cpu/gpu>`
        *   `--py_versions`: `<py2/py3>`

3.  Example: Build all GPU training containers:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

### Upgrading the Framework Version

To upgrade a framework version (e.g., MXNet from 1.6.0 to 1.7.0):

1.  Modify the `buildspec.yml` file to change the version number:
    ```yaml
    # mxnet/training/buildspec.yml
      1   account_id: &ACCOUNT_ID <set-$ACCOUNT_ID-in-environment>
      2   region: &REGION <set-$REGION-in-environment>
      3   framework: &FRAMEWORK mxnet
      4   version: &VERSION 1.6.0 *<--- Change this to 1.7.0*
          ................
    ```
2.  Ensure the corresponding Dockerfile exists in the appropriate directory (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  Build the container as described above.

### Adding Artifacts to Your Build Context

To include files in the build context:

1.  Add the artifact details to the `buildspec.yml` file under the `context`, `training_context`, or `inference_context` key.
    ```yaml
    # mxnet/training/buildspec.yml
     19   context:
     20     README.xyz: *<---- Object name (Can be anything)*
     21       source: README-context.rst *<--- Path for the file to be copied*
     22       target: README.rst *<--- Name for the object in** the build context*
    ```
2.  Build the container as described above.

### Adding a Package

To add a package:

1.  Modify the Dockerfile (e.g., `mxnet/training/docker/1.6.0/py3/Dockerfile.gpu`) to include the package installation command.
    ```dockerfile
    # mxnet/training/docker/1.6.0/py3/Dockerfile.gpu
    139 RUN ${PIP} install --no-cache --upgrade \
    140     keras-mxnet==2.2.4.2 \
    ...........................
    160     awscli \
    161     octopush
    ```
2.  Build the container as described above.

### Running Tests Locally

Run tests locally to iterate faster and avoid resource usage.

1.  **Setup:** Ensure you have the images locally (pull from ECR if necessary) and install test requirements:
    ```shell
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Environment Variables:** Set environment variables for testing:
    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Navigate to Test Directory:**
    ```shell
    cd test/dlc_tests
    ```
4.  **Run Tests (EC2, ECS):**
    ```shell
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto
    ```
    *   Remove `-n=auto` to run tests sequentially.
5.  **Run Tests (EKS):**
    ```shell
    cd ../
    export TEST_TYPE=eks
    python test/testrunner.py
    ```
6.  **Run a Specific Test File:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```
7.  **Run a Specific Test Function:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```
8.  **Run SageMaker Local Mode Tests:**
    *   Launch an EC2 instance with the Deep Learning AMI.
    *   Clone the repository and check out your branch.
    *   Log into the ECR repo.
    *   Navigate to the relevant directory (e.g., `test/sagemaker_tests/mxnet/training/`).
    *   Install requirements.
    *   Run the pytest command, adjusting the paths and parameters as needed:
        ```shell
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
        --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
        --py-version 3
        ```
9.  **Run SageMaker Remote Tests:**
    *   Create an IAM role named "SageMakerRole" with `AmazonSageMakerFullAccess`.
    *   Navigate to the relevant directory.
    *   Run pytest, adjusting paths and parameters:
        ```shell
        pytest integration/sagemaker/test_mnist.py \
        --region us-west-2 --docker-base-name mxnet-training \
        --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
        --instance-type ml.g5.12xlarge
        ```
10. **Run SageMaker Benchmark Tests:**
    *   Create a `sm_benchmark_env_settings.config` file.
    *   Add image URIs, build context, and test type.
    *   Source the settings file.
    *   Run `pip install -r requirements.txt`
    *   Run `python test/testrunner.py` to test all images for multiple frameworks
    *   OR from `deep-learning-containers/` run  `pytest test/dlc_tests/benchmark/<framework-name>/<image-type>/test_*.py` to test one individual framework image type.
    *   The scripts and model-resources used in these tests will be located at:
        ```
        deep-learning-containers/test/dlc_tests/benchmark/sagemaker/<framework-name>/<image-type>/resources/