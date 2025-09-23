# AWS Deep Learning Containers: Build, Train, and Deploy with Ease

Accelerate your machine learning workflows with AWS Deep Learning Containers (DLCs), pre-built Docker images optimized for deep learning frameworks. [Explore the AWS Deep Learning Containers Repository](https://github.com/aws/deep-learning-containers).

**Key Features:**

*   **Optimized Environments:** Pre-configured with popular frameworks like TensorFlow, PyTorch, and MXNet, along with NVIDIA CUDA and Intel MKL libraries.
*   **Amazon ECR Availability:** Easily access and deploy DLCs from the Amazon Elastic Container Registry (ECR).
*   **SageMaker Integration:** Seamlessly integrate with Amazon SageMaker for training, inference, and more.
*   **Broad Compatibility:** Tested and validated for use on Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Customization Options:** Build custom images by adding packages, modifying Dockerfiles, and incorporating your specific dependencies.

## Getting Started

This guide provides steps to build and test the DLCs on platforms such as Amazon SageMaker, EC2, ECS, and EKS.

**Example: Building an MXNet GPU Python 3 Training Container**

1.  **Prerequisites:**
    *   Access to an AWS account, configured with necessary permissions (IAM role recommended). Suggested IAM policies include:
        *   `AmazonEC2ContainerRegistryFullAccess`
        *   `AmazonEC2FullAccess`
        *   `AmazonEKSClusterPolicy`
        *   `AmazonEKSServicePolicy`
        *   `AmazonEKSServiceRolePolicy`
        *   `AWSServiceRoleForAmazonEKSNodegroup`
        *   `AmazonSageMakerFullAccess`
        *   `AmazonS3FullAccess`
    *   An ECR repository (e.g., "beta-mxnet-training" in us-west-2).
    *   Docker installed and configured on your system.

2.  **Configuration:**
    *   Clone the repository.
    *   Set environment variables (replace with your values):

    ```bash
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```

3.  **ECR Login:**

    ```bash
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```

4.  **Setup and Requirements:**
    *   Create a virtual environment and install dependencies:

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

Build your own custom images to match your framework and training requirements.

1.  **Understanding Build Files:** Dockerfiles are located at specific paths (e.g., `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`), defined by the `buildspec.yml` files (e.g., `mxnet/training/buildspec.yml`).

2.  **Local Build:**  Build all images defined in the buildspec.yml:

    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

3.  **Build a Single Image:** Build a specific image, such as a training GPU container with Python 3:

    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

4.  **Build Arguments:** Use `--image_types`, `--device_types`, and `--py_versions` arguments to specify your build. Use comma-separated values for multiple options.

## Upgrading Framework Versions

To incorporate a new framework version, such as MXNet 1.7.0:

1.  **Update `buildspec.yml`:**  Change the `version` key in the buildspec.yml file (e.g., `mxnet/training/buildspec.yml`).

2.  **Create Dockerfile (if needed):** Ensure a corresponding Dockerfile exists (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).

3.  **Rebuild Container:** Follow the building instructions above.

## Adding Artifacts to Your Build Context

1.  **Specify in `buildspec.yml`:**  Add artifacts (files) under the `context`, `training_context`, or `inference_context` keys in your framework's `buildspec.yml` file.

2.  **Build Container:** The specified files will be available within the container during the build process.

## Adding a Package

1.  **Modify Dockerfile:** Add a package install command (e.g., `RUN ${PIP} install --no-cache --upgrade octopush`) to the appropriate Dockerfile.

2.  **Rebuild Container:**  Rebuild the image to include the new package.

## Running Tests Locally

Utilize pytest to perform unit and integration tests, enabling the rapid testing of your images.

1.  **Setup:**

    *   Ensure you have the images you want to test locally (pull from ECR).
    *   Install test requirements:

    ```bash
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```

2.  **Environment Variables:** Set the following variables in your shell:

    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

3.  **Navigate to Tests:**

    ```bash
    cd test/dlc_tests
    ```

4.  **Run Tests:**
    *   Run all tests for a platform (e.g., EC2):

    ```bash
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto
    ```

    *   Run a specific test file:

    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

    *   Run a specific test function:

    ```bash
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

## SageMaker Local Mode Tests

1.  **Prerequisites:**
    *   Launch a CPU or GPU EC2 instance with a Deep Learning AMI.
    *   Clone the repository and check out the correct branch.
    *   Login to the ECR repo.
    *   Navigate to the correct directory.
    *   Install requirements.

2.  **Run Tests:**
    *   Local integration tests (excluding tensorflow\_inference):

        ```bash
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
         --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
         --py-version 3
        ```

    *   tensorflow\_inference py3 images:

        ```bash
        python3 -m  pytest -v integration/local \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference \
        --tag 1.15.2-cpu-py36-ubuntu16.04 --framework-version 1.15.2 --processor cpu
        ```

## SageMaker Remote Tests

1.  **Prerequisites:**

    *   Create an IAM role named "SageMakerRole" with the `AmazonSageMakerFullAccess` policy.
    *   Navigate to the correct directory.
    *   Install requirements.
    *   Run the corresponding pytest command.

2.  **Run Tests:**
    *   Remote integration tests (excluding tensorflow\_inference):

        ```bash
        pytest integration/sagemaker/test_mnist.py \
        --region us-west-2 --docker-base-name mxnet-training \
        --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
        --instance-type ml.g5.12xlarge
        ```

    *   tensorflow\_inference py3 images:

        ```bash
        python3 -m pytest test/integration/sagemaker/test_tfs. --registry {aws_account_id} \
        --region us-west-2  --repo tensorflow-inference --instance-types ml.c5.18xlarge \
        --tag 1.15.2-py3-cpu-build --versions 1.15.2
        ```

## SageMaker Benchmark Tests

1.  **Prerequisites:**

    *   Create `sm_benchmark_env_settings.config` file.
    *   Add export statements for `DLC_IMAGES`, `BUILD_CONTEXT`, `TEST_TYPE`, `CODEBUILD_RESOLVED_SOURCE_VERSION`, and `REGION`.
    *   Source the config file.
    *   Navigate to the correct directory.
    *   Install requirements.
    *   Run pytest or the test runner.
2.  **Run Tests:**
    *   Run test runner for all images:

        ```bash
        pip install -r requirements.txt
        python test/testrunner.py
        ```

    *   Run individual tests:

        ```bash
        cd test/dlc_tests
        pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py
        ```

## Notes

*   SageMaker does not support tensorflow\_inference py2 images.