# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Accelerate your machine learning workflows with AWS Deep Learning Containers, pre-built and optimized Docker images for popular deep learning frameworks.**

**Key Features:**

*   **Pre-built and Optimized:** Get started quickly with pre-configured environments for TensorFlow, TensorFlow 2, PyTorch, and MXNet, optimized for performance on AWS.
*   **GPU and CPU Support:** Leverage the power of GPUs with CUDA support and CPUs with Intel MKL for efficient training and inference.
*   **Seamless Integration:** Designed for use with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Easy Deployment:** Available in Amazon Elastic Container Registry (Amazon ECR) for simple deployment and management.
*   **Flexible Framework Support:** Support for various versions and configurations of the major deep learning frameworks.

[Link to Original Repo](https://github.com/aws/deep-learning-containers)

## Getting Started

Follow these steps to build, test, and deploy AWS Deep Learning Containers. This guide provides instructions for various platforms including Amazon SageMaker, EC2, ECS, and EKS.

### Prerequisites

1.  **AWS Account & Permissions:** Ensure you have access to an AWS account and have set up your environment with the AWS CLI. We recommend using an IAM role. Required IAM policies include:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess

2.  **ECR Repository:** Create an ECR repository in your desired region (e.g., us-west-2).
    *   Example: `aws ecr create-repository --repository-name beta-mxnet-training --region us-west-2`

3.  **Docker:** Ensure you have Docker client set up on your system.

### Building and Deploying a Container (Example: MXNet GPU Python3 Training)

1.  **Clone the Repository:**
    ```bash
    git clone [repository URL]
    cd deep-learning-containers
    ```
2.  **Set Environment Variables:** Replace the placeholders with your account details.
    ```bash
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
3.  **Login to ECR:**
    ```bash
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
    ```
4.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
5.  **Initial Setup:**
    ```bash
    bash src/setup.sh mxnet
    ```

## Building Your Image

1.  **Using `buildspec.yml`:**  The build process uses `buildspec.yml` files located in the framework directories (e.g., `mxnet/training/buildspec.yml`). These files specify the Dockerfile paths.

2.  **Build All Images:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

3.  **Build a Single Image:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
    *   `--image_types`:  `training` or `inference`
    *   `--device_types`: `cpu` or `gpu`
    *   `--py_versions`: `py2` or `py3`

## Upgrading Framework Version

1.  **Modify `buildspec.yml`:** Update the `version` field in the relevant `buildspec.yml` file.

2.  **Update Dockerfile:** Ensure the Dockerfile for the new version exists in the corresponding directory structure (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).

3.  **Build the Container:** Run the build command as described above.

## Adding Artifacts to Your Build Context

1.  **Define Context in `buildspec.yml`:** Specify artifacts under the `context`, `training_context`, or `inference_context` keys in the `buildspec.yml` file.

    ```yaml
    context:
      README.xyz:
        source: README-context.rst
        target: README.rst
    ```

2.  **Build the Container:**  Build the container using the steps described earlier.

## Adding a Package

1.  **Modify the Dockerfile:** Add the package installation command to the Dockerfile (e.g., using `pip install`).

    ```dockerfile
    RUN ${PIP} install --no-cache --upgrade \
        keras-mxnet==2.2.4.2 \
        awscli \
        octopush
    ```

2.  **Build the Container:**  Build the container using the steps described earlier.

## Running Tests Locally

Before submitting a PR, run tests locally to ensure build success and functionality.

1.  **Environment Setup:** Install testing requirements and set environment variables.
    ```bash
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    cd test/dlc_tests
    ```

2.  **Run Tests:**

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

3.  **Run Specific Tests:**
    *   Specify test file: `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py`
    *   Specify test function: `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu`

4. **SageMaker Local Mode Tests:**
    *   Requires EC2 instance with Deep Learning AMI
    *   Clone the repo and check out the branch
    *   Login to ECR
    *   Navigate to appropriate directory: `test/sagemaker_tests/mxnet/training/`
    *   Install requirements: `pip3 install -r requirements.txt`
    *   Run pytest command specifying `--docker-base-name`, `--tag`, and other parameters.

5. **SageMaker Remote Tests:**
    *   Create an IAM role named SageMakerRole
    *   Install requirements
    *   Run pytest command specifying the region, docker base name, AWS ID, instance type and other parameters.

6.  **SageMaker Benchmark Tests:**
    *   Create `sm_benchmark_env_settings.config` and set environment variables (DLC_IMAGES, BUILD_CONTEXT, TEST_TYPE, CODEBUILD_RESOLVED_SOURCE_VERSION, REGION).
    *   Run: `source sm_benchmark_env_settings.config`
    *   Run tests:
        *   For multiple frameworks: `pip install -r requirements.txt; python test/testrunner.py`
        *   For a specific framework: `pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py`

**Note:**  SageMaker does not support tensorflow_inference py2 images.

## License

This project is licensed under the Apache-2.0 License. `smdistributed.dataparallel` and `smdistributed.modelparallel` are released under the [AWS Customer Agreement](https://aws.amazon.com/agreement/).