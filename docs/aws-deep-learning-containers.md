# AWS Deep Learning Containers

**Get pre-built, optimized Docker images for training and serving your machine learning models with AWS Deep Learning Containers (DLCs).** ([Back to Original Repo](https://github.com/aws/deep-learning-containers))

## Key Features

*   **Pre-built and Optimized:**  DLCs provide pre-configured environments with popular deep learning frameworks and optimized libraries, including TensorFlow, PyTorch, MXNet, CUDA, and Intel MKL.
*   **Framework Support:**  Offers support for TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Integration:** Designed for use with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **GPU and CPU Support:**  Available for both GPU and CPU instances, enabling flexibility in your compute choices.
*   **Available in ECR:** The DLCs are readily available in the Amazon Elastic Container Registry (Amazon ECR).

## Getting Started

This section outlines the setup to build and test the DLCs on Amazon SageMaker, EC2, ECS, and EKS.

1.  **Prerequisites:**

    *   An AWS account with appropriate permissions. Ensure your AWS CLI is configured with an IAM user or role with the following managed permissions.
    *   [AmazonEC2ContainerRegistryFullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess)
    *   [AmazonEC2FullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEC2FullAccess)
    *   [AmazonEKSClusterPolicy](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEKSClusterPolicy)
    *   [AmazonEKSServicePolicy](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEKSServicePolicy)
    *   [AmazonEKSServiceRolePolicy](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEKSServiceRolePolicy)
    *   [AWSServiceRoleForAmazonEKSNodegroup](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AWSServiceRoleForAmazonEKSNodegroup)
    *   [AmazonSageMakerFullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonSageMakerFullAccess)
    *   [AmazonS3FullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonS3FullAccess)
    *   An ECR repository created, for example with name “beta-mxnet-training” in the us-west-2 region.
    *   Docker client set up on your system.
2.  **Clone the Repository and Set Environment Variables:**

    ```bash
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
3.  **Login to ECR:**

    ```bash
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
4.  **Create a Virtual Environment and Install Requirements:**

    ```bash
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
5.  **Perform Initial Setup:**

    ```bash
    bash src/setup.sh mxnet
    ```

## Building Your Image

The Dockerfiles are located using the pattern:  `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`.

1.  **Build All Images:**

    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

2.  **Build a Specific Image:**

    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```

    *   Arguments: `--image_types`, `--device_types`, and `--py_versions`  accept comma-separated values.

        *   `--image_types`:  `training`/`inference`
        *   `--device_types`:  `cpu`/`gpu`
        *   `--py_versions`:  `py2`/`py3`
3.  **Example for GPU Training:**

    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

## Upgrading the Framework Version

1.  **Modify `buildspec.yml`:**  Change the framework version within the appropriate `buildspec.yml` file.
2.  **Ensure Dockerfile Exists:**  Confirm the corresponding Dockerfile exists in the correct directory structure.
3.  **Build the Container:**  Follow the instructions in the "Building Your Image" section.

## Adding Artifacts to Your Build Context

1.  **Add to `buildspec.yml`:**  Define artifacts within the `context`, `training_context`, or `inference_context` keys in your framework's `buildspec.yml`.

## Adding a Package

1.  **Modify Dockerfile:** Update the relevant Dockerfile to include the `RUN ${PIP} install` command for your desired package.
2.  **Build the Container:**  Follow the instructions in the "Building Your Image" section.

## Running Tests Locally

1.  **Prerequisites:**
    *   Images to test (pull from ECR).
    *   Install test requirements: `pip install -r src/requirements.txt` and `pip install -r test/requirements.txt`.
    *   Ensure access to a personal/team AWS account.
2.  **Set Environment Variables:**

    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

    \[Note: change the repository name to the one setup in your account]
3.  **Change Directory:**

    ```bash
    cd test/dlc_tests
    ```
4.  **Run All Tests (Series):**

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
5.  **Run a Specific Test File:**

    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```
6.  **Run a Specific Test Function:**

    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

7.  **SageMaker Local Mode Tests:**
    *   Create a CPU or GPU EC2 instance with the latest Deep Learning AMI.
    *   Clone the repository.
    *   Login to ECR.
    *   Change to the appropriate directory for the framework and job type.
    *   Install requirements: `pip3 install -r requirements.txt`
    *   Run tests. Example below for mxnet_training:
    ```bash
    python3 -m pytest -v integration/local --region us-west-2 \
    --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
     --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
     --py-version 3
    ```
8.  **SageMaker Remote Tests:**
    *   Create an IAM role with "SageMakerRole" and the `AmazonSageMakerFullAccess` policy.
    *   Change to the appropriate directory.
    *   Install requirements: `pip3 install -r requirements.txt`
    *   Run the appropriate pytest command for the tests you want to run.
    *   Example below for mxnet_training:
    ```bash
    pytest integration/sagemaker/test_mnist.py \
    --region us-west-2 --docker-base-name mxnet-training \
    --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
    --instance-type ml.g5.12xlarge
    ```
9.  **SageMaker Benchmark Tests:**
    *   Create `sm_benchmark_env_settings.config`.
    *   Set environment variables in the file.
    *   Run `source sm_benchmark_env_settings.config`.
    *   Test all images: `pip install -r requirements.txt` and `python test/testrunner.py`.
    *   Test a single image:  Run the pytest command in test/dlc_tests

    *   The scripts and model-resources used in these tests will be located at:
        ```
        deep-learning-containers/test/dlc_tests/benchmark/sagemaker/<framework-name>/<image-type>/resources/
        ```
    *   Note: SageMaker does not support tensorflow_inference py2 images.