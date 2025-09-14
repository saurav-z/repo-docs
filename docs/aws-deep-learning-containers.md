# AWS Deep Learning Containers: Build, Train, and Deploy Machine Learning Models

**Accelerate your machine learning workflows with AWS Deep Learning Containers (DLCs), optimized Docker images for popular deep learning frameworks.** ([Original Repository](https://github.com/aws/deep-learning-containers))

## Key Features

*   **Optimized Environments:** Pre-built Docker images with TensorFlow, TensorFlow 2, PyTorch, and MXNet, including optimized libraries for GPU (CUDA) and CPU (Intel MKL) instances.
*   **Broad Compatibility:** Works seamlessly with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Easy Deployment:** Available in Amazon Elastic Container Registry (Amazon ECR) for streamlined deployment.
*   **Framework Support:** Supports a wide range of framework versions for training and inference.

## Getting Started

This section provides instructions for building, testing, and using the AWS Deep Learning Containers.

### Prerequisites

1.  **AWS Account and Permissions:** Ensure you have an active AWS account. Set up your AWS CLI and configure access with the appropriate IAM permissions. Recommended IAM policies include:

    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
2.  **ECR Repository:** Create an ECR repository (e.g., "beta-mxnet-training") in your desired AWS region (e.g., us-west-2).
3.  **Docker:** Install and configure the Docker client on your system.

### Setup

1.  **Clone the Repository:** Clone the `aws/deep-learning-containers` repository.

2.  **Set Environment Variables:** Set the following environment variables:
    ```bash
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```

3.  **Login to ECR:** Authenticate with ECR:
    ```bash
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```

4.  **Create Virtual Environment and Install Dependencies:**
    ```bash
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```

5.  **Initial Setup:** Run the setup script for a specific framework:
    ```bash
    bash src/setup.sh mxnet
    ```

### Building Your Image

1.  **Understanding Dockerfile Paths:** Dockerfiles are located in a structured directory, e.g., `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`.

2.  **Building All Images:** Use the following command to build all images specified in the buildspec.yml:
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

3.  **Building a Single Image:** Build a specific image by specifying the desired parameters:
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
    The arguments `--image_types`, `--device_types` and `--py_versions` accept comma separated lists of their values.

4.  **Upgrading the Framework Version:**
    *   Modify the `version` key in the relevant `buildspec.yml` file to the new framework version.
    *   Ensure the corresponding Dockerfile exists in the correct directory structure.

5.  **Adding Artifacts to Your Build Context:**
    *   Define artifacts in the `buildspec.yml` file under the `context`, `training_context`, or `inference_context` keys.

6.  **Adding a Package:**
    *   Modify the Dockerfile to include the `pip install` command for the desired package.

### Running Tests Locally

Test your changes locally to save on resources and speed up development.

1.  **Prerequisites:** Ensure the images you want to test are available locally (e.g., pulled from ECR). Install test requirements.
    ```bash
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Set Environment Variables:**
    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

3.  **Navigate to Test Directory:**
    ```bash
    cd test/dlc_tests
    ```

4.  **Running Tests:**

    *   **EC2:**
        ```bash
        pytest -s -rA ec2/ -n=auto
        ```
    *   **ECS:**
        ```bash
        pytest -s -rA ecs/ -n=auto
        ```
    *   **EKS:**
        ```bash
        cd ../
        export TEST_TYPE=eks
        python test/testrunner.py
        ```
    *   **Specific Test File:**
        ```bash
        pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
        ```
    *   **Specific Test Function:**
        ```bash
        pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
        ```
    *   **SageMaker Local Mode:** Follow the detailed instructions in the original README.
    *   **SageMaker Remote Tests:**  Follow the detailed instructions in the original README.
    *   **SageMaker Benchmark Tests:**  Follow the detailed instructions in the original README.