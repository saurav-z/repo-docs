# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started with pre-built, optimized Docker images for training and serving your machine learning models on AWS.**  [View the original repository](https://github.com/aws/deep-learning-containers)

AWS Deep Learning Containers (DLCs) provide a convenient way to leverage popular deep learning frameworks on Amazon Web Services.  These Docker images are pre-configured with optimized environments, making it easy to train and deploy your models efficiently.

**Key Features:**

*   **Optimized Frameworks:** Built-in support for TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **GPU & CPU Support:**  Includes NVIDIA CUDA for GPU instances and Intel MKL for CPU instances.
*   **Amazon ECR Availability:** Easily accessible within the Amazon Elastic Container Registry (ECR).
*   **SageMaker Integration:** Seamlessly used with Amazon SageMaker for training, inference, and other jobs.
*   **Broad Compatibility:** Tested and validated for use on Amazon EC2, Amazon ECS, and Amazon EKS.

## Getting Started

This section provides instructions for building and testing the DLCs, using an example with MXNet GPU python3 training container.

**Prerequisites:**

*   Access to an AWS account with the appropriate permissions (IAM role recommended).
*   [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) configured.
*   [Docker](https://docs.docker.com/get-docker/) installed and running.

**Setup Steps:**

1.  **AWS Account Setup:** Configure your AWS CLI with an IAM role that includes the following managed permissions:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess

2.  **Create ECR Repository:** Create an ECR repository named "beta-mxnet-training" in the `us-west-2` region.

3.  **Clone and Configure:**
    *   Clone the repository:

    *   Set environment variables:
    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
    *   Login to ECR:
    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```

4.  **Setup Virtual Environment and Install Dependencies:**
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```

5.  **Initial Setup:**
    ```shell
    bash src/setup.sh mxnet
    ```

## Building Your Image

The Dockerfiles are organized by framework, training/inference type, and other parameters, defined in the `buildspec.yml` files.

1.  **Build All Images (as defined in buildspec.yml):**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

2.  **Build a Specific Image:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```

3.  **Image Type, Device, and Python Version Parameters:**
    *   `--image_types`:  `<training/inference>`
    *   `--device_types`: `<cpu/gpu>`
    *   `--py_versions`: `<py2/py3>`

4.  **Example: Build GPU Training Containers:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

## Upgrading the Framework Version

Instructions for updating the framework version (e.g., MXNet) can be found in the original README.

1.  **Modify `buildspec.yml`:** Update the `version` parameter within the `buildspec.yml` file for the desired framework (e.g., MXNet) to the new version number (e.g., 1.7.0).
2.  **Ensure Dockerfile Path:** Confirm the Dockerfile exists at the expected path, which is determined by the version, processor type and python version.
3.  **Build the container** using the instructions detailed above.

## Adding Artifacts to Your Build Context

This outlines how to include additional files or resources within the build context of your Docker images.

1.  **Context Configuration:** Add the file in the framework buildspec file under the context key:
    ```yaml
     19 context:
     20   README.xyz: *<---- Object name (Can be anything)*
     21     source: README-context.rst *<--- Path for the file to be copied*
     22     target: README.rst *<--- Name for the object in** the build context*
    ```
2.  **Context Options:**
    *   `context`: Add the file to all images
    *   `training_context`: Add the file only for training images
    *   `inference_context`: Add the file only for inference images
3.  **For single container:** If you need it for a single container add it under the context key for that particular image.

## Adding a Package

This section describes how to add packages to the DLC.
1.  **Modify Dockerfile:**  Edit the Dockerfile (e.g., `/mxnet/training/docker/1.6.0/py3/Dockerfile.gpu`) to include the `octopush` package, ensuring the package is installed:
    ```dockerfile
    161     octopush
    ```
2.  **Rebuild the image** using the instructions detailed above.

## Running Tests Locally

Detailed instructions for running tests locally, including setting up the testing environment and running tests on different platforms (EC2, ECS, EKS, and SageMaker), are available in the original README.

**Important: These tests require access to an AWS account.**

1.  **Environment Setup:**
    *   Install testing requirements:
        ```shell
        pip install -r src/requirements.txt
        pip install -r test/requirements.txt
        ```
    *   Define the images and version to test:
        ```shell
        export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
        export PYTHONPATH=$(pwd)/src
        export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
        ```
    *   Change the working directory to the test directory for the specific framework.
2.  **Running Tests:**
    *   **EC2:** `pytest -s -rA ec2/ -n=auto`
    *   **ECS:** `pytest -s -rA ecs/ -n=auto`
    *   **EKS:**  `cd ../; export TEST_TYPE=eks; python test/testrunner.py`
    *   **Specific Test File:** `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py`
    *   **Specific Test Function:** `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu`
    *   **SageMaker Local Mode:**  Run the commands detailed in the original README.
    *   **SageMaker Remote:** Follow the steps in the original README.
    *   **SageMaker Benchmark:** Follow the steps in the original README.

**Note:**  SageMaker does not support tensorflow\_inference py2 images.