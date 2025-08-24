# AWS Deep Learning Containers: Build, Train, and Deploy Your ML Models

**Accelerate your machine learning workflows with AWS Deep Learning Containers (DLCs), pre-built Docker images optimized for training and serving deep learning models.**  ([Original Repo](https://github.com/aws/deep-learning-containers))

## Key Features:

*   **Optimized Environments:** DLCs provide pre-configured environments with popular deep learning frameworks like TensorFlow, TensorFlow 2, PyTorch, and MXNet, pre-installed with optimized libraries such as NVIDIA CUDA and Intel MKL.
*   **Amazon SageMaker Integration:**  Seamlessly integrates with Amazon SageMaker for training, inference, and other job types.
*   **Flexible Deployment:**  Tested and compatible with Amazon EC2, Amazon ECS, and Amazon EKS for versatile deployment options.
*   **Pre-built and Ready to Use:** Available in Amazon Elastic Container Registry (Amazon ECR), ready for immediate use.
*   **Simplified Development:** Streamlines the development process by providing a consistent and optimized environment, reducing the need for manual configuration.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the Framework Version](#upgrading-the-framework-version)
*   [Adding Artifacts to Your Build Context](#adding-artifacts-to-your-build-context)
*   [Adding a Package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

### Getting Started

This section outlines how to set up your environment to build and test the DLCs on Amazon SageMaker, EC2, ECS, and EKS.

**Prerequisites:**

*   An active AWS account with appropriate permissions.  We recommend using an IAM role with the following managed policies (or equivalent permissions):
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
*   [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) configured to access your account.
*   [Docker](https://docs.docker.com/get-docker/) installed and configured.

**Example: Building an MXNet GPU Python3 Training Container:**

1.  **Clone the repository** and set the following environment variables:

    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```

2.  **Create an ECR repository** (if you haven't already):

    ```shell
    aws ecr create-repository --repository-name beta-mxnet-training --region us-west-2
    ```

3.  **Login to ECR:**

    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```

4.  **Create and activate a virtual environment** and install the required packages:

    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```

5.  **Perform the initial setup:**

    ```shell
    bash src/setup.sh mxnet
    ```

### Building Your Image

The Dockerfiles are located in a specific structure, following a pattern:  `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`.  The `buildspec.yml` files (e.g., `mxnet/training/buildspec.yml`) define the build configurations for each framework and image type.

1.  **Build all images defined in the buildspec.yml:**

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

2.  **Build a single image:**

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet --image_types training --device_types cpu --py_versions py3
    ```

    *   Use the `--image_types`, `--device_types`, and `--py_versions` arguments to specify image parameters.  Valid values are:
        *   `--image_types`:  `training`, `inference`
        *   `--device_types`: `cpu`, `gpu`
        *   `--py_versions`: `py2`, `py3`

### Upgrading the Framework Version

To update the framework version, you'll need to modify the corresponding `buildspec.yml` file and ensure the Dockerfile for the new version exists.

1.  **Modify `buildspec.yml`:**  Update the `version` key (e.g., from `1.6.0` to `1.7.0`).

    ```yaml
    # mxnet/training/buildspec.yml
    # ...
    version: &VERSION 1.7.0  # <--- Change this
    # ...
    ```

2.  **Verify Dockerfile Location:** Ensure the Dockerfile for the new version exists at the correct path.

    ```
    mxnet/training/docker/1.7.0/py3/Dockerfile.gpu  # Example path
    ```

3.  **Build the container** as described above.

### Adding Artifacts to Your Build Context

To include files (artifacts) from your build context within the image, you can leverage the `context` sections in `buildspec.yml` files. This allows you to copy custom files into the image during the build process.

1.  **Add the artifact to the build context** using the `context` key in the `buildspec.yml` file. You can add files to all images, training images, inference images, or specific images.

    ```yaml
    # mxnet/training/buildspec.yml
    context:
      README.xyz:   # Object name (can be anything)
        source: README-context.rst # Path for the file to be copied
        target: README.rst        # Name for the object in the build context
    ```

    Alternatively, you can use:

    ```yaml
    training_context: &TRAINING_CONTEXT
      README.xyz:
        source: README-context.rst
        target: README.rst
    ```

    or

    ```yaml
      images:
        BuildMXNetCPUTrainPy3DockerImage:
          # ...
          context:
            <<: *TRAINING_CONTEXT
            README.xyz:
              source: README-context.rst
              target: README.rst
    ```
2. **Build the container** as described above.

### Adding a Package

Customize your containers by adding packages to the Dockerfile.

1.  **Modify the Dockerfile** to include the `pip install` command for the desired package.

    ```dockerfile
    # mxnet/training/docker/1.6.0/py3/Dockerfile.gpu
    # ...
    RUN ${PIP} install --no-cache --upgrade \
        keras-mxnet==2.2.4.2 \
    # ...
        awscli \
        octopush  # <--- Add the package here
    ```

2.  **Build the container** as described above.

### Running Tests Locally

Test your images locally to avoid waiting for a build to complete or using extra resources.

1.  **Prerequisites:** Ensure the images you want to test are available locally (pull them from ECR if needed).  Install test requirements.
    ```shell
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```

2.  **Set environment variables:**  Define `DLC_IMAGES`, `CODEBUILD_RESOLVED_SOURCE_VERSION`, and `PYTHONPATH`.

    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

3.  **Navigate to the test directory:**

    ```shell
    cd test/dlc_tests
    ```

4.  **Run all tests (in series) for a platform:**

    ```shell
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto
    # EKS
    cd ../
    export TEST_TYPE=eks
    python test/testrunner.py
    ```

    *   Remove `-n=auto` to run tests sequentially.

5.  **Run a specific test file:**

    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

6.  **Run a specific test function:**

    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

7.  **Run SageMaker local mode tests:**
    1.  Launch a CPU or GPU EC2 instance with the latest Deep Learning AMI.
    2.  Clone your GitHub branch and run the login and install commands.
    3.  Navigate to the appropriate test directory (`test/sagemaker_tests/{framework}/{job_type}`).
    4. Run the pytest command for local mode tests.

        ```shell
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
        --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
        --py-version 3
        ```
    5. For Tensorflow Inference Tests, use the command

    ```shell
    python3 -m  pytest -v integration/local \
    --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference \
    --tag 1.15.2-cpu-py36-ubuntu16.04 --framework-version 1.15.2 --processor cpu
    ```
8. **Run SageMaker remote tests:**

    1. Create an IAM role named "SageMakerRole" with the `AmazonSageMakerFullAccess` managed policy.
    2.  Navigate to the appropriate test directory  (`test/sagemaker_tests/{framework}/{job_type}`).
    3.  Run the pytest command for remote tests.

    ```shell
    pytest integration/sagemaker/test_mnist.py \
    --region us-west-2 --docker-base-name mxnet-training \
    --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
    --instance-type ml.g5.12xlarge
    ```

    4. For Tensorflow Inference Tests, use the command

    ```shell
      python3 -m pytest test/integration/sagemaker/test_tfs. --registry {aws_account_id} \
      --region us-west-2  --repo tensorflow-inference --instance-types ml.c5.18xlarge \
      --tag 1.15.2-py3-cpu-build --versions 1.15.2
    ```
9.  **Run SageMaker benchmark tests:**
    1.  Create `sm_benchmark_env_settings.config` file in the `deep-learning-containers/` folder and export the necessary environment variables.
    2.  Run the benchmark tests.
        ```shell
        pip install -r requirements.txt
        python test/testrunner.py