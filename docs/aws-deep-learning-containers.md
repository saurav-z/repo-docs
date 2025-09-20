## AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started quickly with pre-built, optimized Docker images for deep learning, making it easier to train and deploy your machine learning models on AWS.**  ([Original Repository](https://github.com/aws/deep-learning-containers))

AWS Deep Learning Containers (DLCs) are pre-configured Docker images designed for training and serving models using popular deep learning frameworks. They provide a streamlined environment with optimized libraries for TensorFlow, TensorFlow 2, PyTorch, and MXNet, along with NVIDIA CUDA (for GPU instances) and Intel MKL (for CPU instances).  Available in Amazon Elastic Container Registry (ECR), these containers are the default for Amazon SageMaker jobs and are also tested for use on Amazon EC2, ECS, and EKS.

**Key Features:**

*   **Pre-built and Optimized:** Ready-to-use images with pre-installed frameworks, libraries, and dependencies.
*   **Framework Support:**  Includes TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Hardware Acceleration:** Optimized for both CPU and GPU instances with Intel MKL and NVIDIA CUDA.
*   **Amazon Integration:**  Seamlessly integrates with Amazon SageMaker, EC2, ECS, and EKS.
*   **Available in ECR:** Easily accessible from the Amazon Elastic Container Registry.

### Available Images

For a comprehensive list of available DLC images, see [Available Deep Learning Containers Images](available_images.md) and find more information on the images available in Sagemaker [here](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html)

### Getting Started

This section guides you through building and testing the DLCs on platforms like Amazon SageMaker, EC2, ECS, and EKS.

#### Prerequisites

*   An AWS account.
*   AWS CLI configured with the necessary permissions (IAM role recommended):
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
*   Docker client setup.

#### Example: Building and Running MXNet GPU Python3 Training Container

1.  **Set up your environment:**
    *   Create an ECR repository (e.g., "beta-mxnet-training" in us-west-2).
    *   Clone the repository.
    *   Set environment variables:

    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```

2.  **Authenticate with ECR:**

    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```

3.  **Install dependencies:**

    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```

4.  **Perform initial setup:**

    ```shell
    bash src/setup.sh mxnet
    ```

### Building Your Image

Dockerfiles are structured with a specific pattern (e.g., `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`).  The `buildspec.yml` files (e.g., `mxnet/training/buildspec.yml`) define these paths.  Modify the `buildspec.yml` to build specific versions or introduce new framework versions.

1.  **Build all images defined in `buildspec.yml`:**

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

2.  **Build a single image:**

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```

3.  **Build parameters:**

    *   `--image_types`:  `training` or `inference`
    *   `--device_types`:  `cpu` or `gpu`
    *   `--py_versions`: `py2` or `py3`

4.  **Example: Build all GPU training containers:**

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

### Upgrading the Framework Version

To upgrade a framework version (e.g., MXNet to 1.7.0):

1.  **Modify `buildspec.yml`:** Change the version parameter (e.g., `version: &VERSION 1.7.0`).

2.  **Ensure the Dockerfile exists:**  Verify that the appropriate Dockerfile exists (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).

3.  **Build the container** as described above

### Adding Artifacts to Your Build Context

1.  **Add the artifact to `buildspec.yml` under the `context` key:**

    ```yaml
    context:
      README.xyz:  # Object name
        source: README-context.rst  # Source file path
        target: README.rst  # Target file name in build context
    ```

2.  You can use `training_context` or `inference_context` to restrict context to training/inference images.

3.  For a single container, add under the context key for that specific image.

4.  **Build the container** as described above

### Adding a Package

1.  **Modify the Dockerfile** (e.g., add `octopush` to the `RUN pip install ...` command).

    ```dockerfile
    RUN ${PIP} install --no-cache --upgrade \
        keras-mxnet==2.2.4.2 \
        ...
        awscli \
        octopush
    ```

2.  **Build the container** as described above.

### Running Tests Locally

Run tests locally to save resources and iterate quickly.  Requires access to an AWS account and the images you want to test.

1.  **Prerequisites:**

    *   Ensure the images are available locally (pull from ECR).
    *   Install test requirements:

    ```shell
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```

2.  **Set environment variables:**

    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
    *Note: Replace `us-west-2` with your desired AWS region*
3.  **Change Directory:**

    ```shell
    cd test/dlc_tests
    ```

4.  **Run tests (EC2/ECS):**
    ```shell
    # EC2 (Parallel)
    pytest -s -rA ec2/ -n=auto
    # ECS (Parallel)
    pytest -s -rA ecs/ -n=auto
    ```
    *Remove `-n=auto` to run tests sequentially*
    ```shell
    #EKS
    cd ../
    export TEST_TYPE=eks
    python test/testrunner.py
    ```

5.  **Run a specific test file:**

    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

6.  **Run a specific test function:**

    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

7.  **SageMaker Local Mode Tests:**

    *   Launch a CPU or GPU EC2 instance with the latest Deep Learning AMI.
    *   Clone the repository.
    *   Login into the ECR repo where the new docker images built exist
        ```shell
        $(aws ecr get-login --no-include-email --registry-ids ${aws_id} --region ${aws_region})
        ```
    *   Navigate to the appropriate test directory (e.g.,  `test/sagemaker_tests/mxnet/training/`).
    *   Install requirements.
    *   Run the tests:
        ```shell
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
         --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
         --py-version 3
        ```

8.  **SageMaker Remote Tests:**

    *   Create an IAM role named "SageMakerRole" with the `AmazonSageMakerFullAccess` policy.
    *   Navigate to the appropriate test directory
        ```shell
        cd test/sagemaker_tests/mxnet/training/
        ```
    *   Run the tests (e.g. for mxnet training):
        ```shell
        pytest integration/sagemaker/test_mnist.py \
        --region us-west-2 --docker-base-name mxnet-training \
        --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
        --instance-type ml.g5.12xlarge
        ```
9.  **SageMaker Benchmark Tests:**

    *   Create `sm_benchmark_env_settings.config` in the project root:

        ```shell
        export DLC_IMAGES="<image_uri_1-you-want-to-benchmark-test>"
        # export DLC_IMAGES="$DLC_IMAGES <image_uri_2-you-want-to-benchmark-test>"
        # export DLC_IMAGES="$DLC_IMAGES <image_uri_3-you-want-to-benchmark-test>"
        export BUILD_CONTEXT=PR
        export TEST_TYPE=benchmark-sagemaker
        export CODEBUILD_RESOLVED_SOURCE_VERSION=$USER
        export REGION=us-west-2
        ```

    *   Run:
        ```shell
        source sm_benchmark_env_settings.config
        ```
    *   Run all images:
        ```shell
        pip install -r requirements.txt
        python test/testrunner.py
        ```
    *   Run a specific framework image type:
        ```shell
        # Assuming that the cwd is deep-learning-containers/
        cd test/dlc_tests
        pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py
        ```

    *   Benchmark scripts and model resources are in  `deep-learning-containers/test/dlc_tests/benchmark/sagemaker/<framework-name>/<image-type>/resources/`.