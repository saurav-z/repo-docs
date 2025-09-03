# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started with pre-built, optimized Docker images for training and serving your machine learning models using TensorFlow, PyTorch, and MXNet.** ([Original Repo](https://github.com/aws/deep-learning-containers))

## Key Features:

*   **Pre-built and Optimized:** Ready-to-use Docker images with pre-installed frameworks (TensorFlow, PyTorch, MXNet), NVIDIA CUDA (for GPU), and Intel MKL (for CPU).
*   **Framework Support:**  Images available for popular deep learning frameworks like TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Amazon Integration:** Seamlessly integrates with Amazon SageMaker, EC2, ECS, and EKS for training, inference, and more.
*   **Easy Deployment:** Available in Amazon Elastic Container Registry (Amazon ECR) for easy deployment.
*   **Flexible Testing:** Supports local and remote testing on Amazon EC2, ECS, and SageMaker for efficient development and debugging.

## Contents

*   [Getting Started](#getting-started)
*   [Building your Image](#building-your-image)
*   [Upgrading the framework version](#upgrading-the-framework-version)
*   [Adding artifacts to your build context](#adding-artifacts-to-your-build-context)
*   [Adding a package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

## Getting Started

Follow these steps to build and test the DLCs on Amazon SageMaker, EC2, ECS, and EKS.

**Example: Building an MXNet GPU Python3 Training Container**

1.  **Prerequisites:**
    *   An AWS account with configured AWS CLI access (IAM user or role) and the following managed policies:
        *   AmazonEC2ContainerRegistryFullAccess
        *   AmazonEC2FullAccess
        *   AmazonEKSClusterPolicy
        *   AmazonEKSServicePolicy
        *   AmazonEKSServiceRolePolicy
        *   AWSServiceRoleForAmazonEKSNodegroup
        *   AmazonSageMakerFullAccess
        *   AmazonS3FullAccess
    *   Docker client installed.
    *   Create an ECR repository (e.g., "beta-mxnet-training" in us-west-2).

2.  **Setup:**
    *   Clone the repository.
    *   Set environment variables: `ACCOUNT_ID`, `REGION`, `REPOSITORY_NAME`.
    *   Login to ECR: `aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com`.
    *   Create a virtual environment and install requirements: `python3 -m venv dlc`, `source dlc/bin/activate`, `pip install -r src/requirements.txt`.
    *   Perform the initial setup: `bash src/setup.sh mxnet`.

## Building your Image

Build your custom Docker images using the provided build scripts.

1.  **Building all images:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

2.  **Building a single image:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
    *   Use `--image_types`, `--device_types`, and `--py_versions` arguments to specify image configurations (e.g., training/inference, cpu/gpu, py2/py3).

## Upgrading the framework version

1.  Modify the `version` in the `buildspec.yml` file.
2.  Ensure the Dockerfile for the new version exists.
3.  Build the container.

## Adding artifacts to your build context

1.  Add artifacts under the `context` key within the `buildspec.yml` file.
2.  You can add them for all images, training/inference images, or a specific container.
3.  Build the container.

## Adding a package

1.  Modify the Dockerfile to install the desired package.
2.  Build the container.

## Running Tests Locally

Test your images locally to save resources and time.

1.  **Setup:**
    *   Ensure you have the images you want to test (pull from ECR if needed).
    *   Install the test requirements: `pip install -r src/requirements.txt`, `pip install -r test/requirements.txt`.
    *   Set environment variables: `DLC_IMAGES`, `PYTHONPATH`, and `CODEBUILD_RESOLVED_SOURCE_VERSION`.
    *   Change the directory to `test/dlc_tests`.

2.  **Running Tests:**

    *   **EC2:** `pytest -s -rA ec2/ -n=auto`
    *   **ECS:** `pytest -s -rA ecs/ -n=auto`
    *   **EKS:**
        ```shell
        cd ../
        export TEST_TYPE=eks
        python test/testrunner.py
        ```
    *   **Specific Test File:** `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py`
    *   **Specific Test Function:** `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu`
    *   **SageMaker local mode:**
        *   Clone your github branch with changes.
        *   Login into the ECR repo where the new docker images built exist.
        *   Change to the appropriate directory (sagemaker_tests/{framework}/{job_type}).
        *   Run the pytest command to test.
    *   **SageMaker remote tests:**
        *   Create an IAM role with name “SageMakerRole”.
        *   Change to the appropriate directory (sagemaker_tests/{framework}/{job_type}).
        *   Run the pytest command to test.
    *   **SageMaker benchmark tests:**
        *   Create a file named `sm_benchmark_env_settings.config` in the deep-learning-containers/ folder.
        *   Run the setup.
        *   To test all images for multiple frameworks, run:
            *   Run `pip install -r requirements.txt` and `python test/testrunner.py`
        *   To test one individual framework image type, run:
            *   Run `pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py`