# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Accelerate your machine learning workflows with AWS Deep Learning Containers (DLCs), pre-configured Docker images for training and serving models on AWS.** ([Original Repository](https://github.com/aws/deep-learning-containers))

## Key Features:

*   **Optimized Environments:** Pre-built with popular deep learning frameworks like TensorFlow, PyTorch, and MXNet, along with NVIDIA CUDA and Intel MKL libraries.
*   **Framework Support:**  Ready-to-use containers for TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Integration with AWS Services:** Seamlessly deploy and scale your models on Amazon SageMaker, EC2, ECS, and EKS.
*   **Available on ECR:** Easily accessible images in the Amazon Elastic Container Registry (ECR).

## Getting Started

This section provides instructions for setting up your environment and building the MXNet GPU python3 training container as an example.

### Prerequisites:

*   An active AWS account. Configure your AWS CLI with an IAM user or role. We recommend using an IAM role with the following managed permissions:
    *   `AmazonEC2ContainerRegistryFullAccess`
    *   `AmazonEC2FullAccess`
    *   `AmazonEKSClusterPolicy`
    *   `AmazonEKSServicePolicy`
    *   `AmazonEKSServiceRolePolicy`
    *   `AWSServiceRoleForAmazonEKSNodegroup`
    *   `AmazonSageMakerFullAccess`
    *   `AmazonS3FullAccess`
*   Docker client installed on your system.

### Steps:

1.  **Set up your AWS account:** Access to an AWS account and properly configure your environment.
2.  **Create ECR Repository:** Create an ECR repository named "beta-mxnet-training" in the `us-west-2` region.
3.  **Clone the Repository:** Clone the `aws/deep-learning-containers` repository.
4.  **Set Environment Variables:**
    ```bash
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
5.  **Login to ECR:**
    ```bash
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
6.  **Create a Virtual Environment & Install Dependencies:**
    ```bash
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
7.  **Initial Setup:**
    ```bash
    bash src/setup.sh mxnet
    ```

## Building Your Image

The build process uses `buildspec.yml` files located in the framework-specific directories (e.g., `mxnet/training/buildspec.yml`).

1.  **Build All Images (Defined in buildspec.yml):**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
    The first run may take longer as it downloads base layers.
2.  **Build a Single Image:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
3.  **Available Build Options:**
    *   `--image_types`: `training` or `inference`
    *   `--device_types`: `cpu` or `gpu`
    *   `--py_versions`: `py2` or `py3`

### Upgrading the Framework Version

1.  Modify the `version` key in the `buildspec.yml` file for the target framework.  For instance, to upgrade MXNet from 1.6.0 to 1.7.0, adjust the `mxnet/training/buildspec.yml`:
    ```yaml
    # mxnet/training/buildspec.yml
      4   version: &VERSION 1.7.0
    ```
2.  Ensure the Dockerfile for the new version exists (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  Rebuild the container using the steps above.

### Adding Artifacts to Your Build Context

Add files to your build context by including them in the `context` section of the `buildspec.yml` file.

1.  In `buildspec.yml`, use the `context` section to copy files into the build context. The following example copies `README-context.rst` to `README.rst`:

    ```yaml
     19 context:
     20   README.xyz:
     21     source: README-context.rst
     22     target: README.rst
    ```
2.  You can define `training_context` or `inference_context` to limit to specific image types, or add it under the context key for that particular image.
3.  Rebuild the container.

### Adding a Package

1.  Modify the Dockerfile to install the required package. For example, to install `octopush` in an MXNet 1.6.0 py3 GPU image, add it to the Dockerfile.
    ```dockerfile
    139 RUN ${PIP} install --no-cache --upgrade \
    140     keras-mxnet==2.2.4.2 \
    ...........................
    160     awscli \
    161     octopush
    ```
2.  Rebuild the container.

## Running Tests Locally

Test your containers locally using `pytest`.

1.  **Setup:**
    *   Ensure you have the images you want to test available locally (pull from ECR if needed).
    *   Install test requirements:
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
3.  **Change Directory:**
    ```bash
    cd test/dlc_tests
    ```
4.  **Run Tests (All for a Platform):**
    ```bash
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto
    # EKS
    cd ../
    export TEST_TYPE=eks
    python test/testrunner.py
    ```
5.  **Run a Specific Test File:**
    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```
6.  **Run a Specific Test Function:**
    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

7.  **SageMaker Local Mode Tests:**
    *   Launch an EC2 instance with the latest Deep Learning AMI.
    *   Clone your GitHub branch.
    *   Login to ECR.
    *   Change to the appropriate directory (e.g., `test/sagemaker_tests/mxnet/training/`).
    *   Install requirements: `pip3 install -r requirements.txt`
    *   Run tests:
        ```bash
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
         --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
         --py-version 3
        ```
        or (for tensorflow_inference py3 images)
        ```bash
        python3 -m  pytest -v integration/local \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference \
        --tag 1.15.2-cpu-py36-ubuntu16.04 --framework-version 1.15.2 --processor cpu
        ```

8.  **SageMaker Remote Tests:**
    *   Create an IAM role named "SageMakerRole" with `AmazonSageMakerFullAccess`.
    *   Change to the test directory.
    *   Install requirements: `pip3 install -r requirements.txt`
    *   Run tests:
        ```bash
        pytest integration/sagemaker/test_mnist.py \
        --region us-west-2 --docker-base-name mxnet-training \
        --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
        --instance-type ml.g5.12xlarge
        ```
        or (for tensorflow_inference py3 images)
        ```bash
        python3 -m pytest test/integration/sagemaker/test_tfs. --registry {aws_account_id} \
        --region us-west-2  --repo tensorflow-inference --instance-types ml.c5.18xlarge \
        --tag 1.15.2-py3-cpu-build --versions 1.15.2
        ```

9.  **SageMaker Benchmark Tests:**
    *   Create `sm_benchmark_env_settings.config` and set environment variables.
    *   Run tests.

Note: SageMaker does not support tensorflow_inference py2 images.