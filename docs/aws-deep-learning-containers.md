# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Accelerate your machine learning workflows with AWS Deep Learning Containers, pre-built and optimized Docker images for popular deep learning frameworks.** ([Original Repo](https://github.com/aws/deep-learning-containers))

## Key Features

*   **Pre-built and Optimized:** Includes TensorFlow, TensorFlow 2, PyTorch, and MXNet with optimized libraries for CPU and GPU instances, including NVIDIA CUDA and Intel MKL.
*   **Amazon ECR Availability:** Easily access and deploy images from Amazon Elastic Container Registry (Amazon ECR).
*   **Seamless Integration:** Designed for use with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Framework Support:**  Offers support for major deep learning frameworks and their respective versions.

## Getting Started

This section guides you through the setup process for building and testing AWS Deep Learning Containers (DLCs) on platforms like Amazon SageMaker, EC2, ECS, and EKS.

**Prerequisites:**

*   An AWS account with appropriate permissions.  Recommended IAM role with these managed policies:  `AmazonEC2ContainerRegistryFullAccess`, `AmazonEC2FullAccess`, `AmazonEKSClusterPolicy`, `AmazonEKSServicePolicy`, `AmazonEKSServiceRolePolicy`, `AWSServiceRoleForAmazonEKSNodegroup`, `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`.
*   Docker client installed and configured on your system.

**Example: Building an MXNet GPU Python3 Training Container**

1.  **Set up your environment:**
    *   Clone the repository.
    *   Set environment variables: `ACCOUNT_ID`, `REGION`, `REPOSITORY_NAME`.
    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
2.  **Authenticate with ECR:**
    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
3.  **Create and activate a virtual environment:**
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
4.  **Perform initial setup:**
    ```shell
    bash src/setup.sh mxnet
    ```

## Building Your Image

The Dockerfiles are located following a specific pattern, e.g., `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`.  Modify `buildspec.yml` in the relevant framework directory to customize your builds.

1.  **Build all images:**
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
3.  **Available Arguments:**
    *   `--image_types`: `<training/inference>` (comma-separated list)
    *   `--device_types`: `<cpu/gpu>` (comma-separated list)
    *   `--py_versions`: `<py2/py3>` (comma-separated list)

## Upgrading the Framework Version

1.  Update the `version` in the buildspec.yml file.
    ```yaml
    # mxnet/training/buildspec.yml
      4   version: &VERSION 1.6.0 *<--- Change this to 1.7.0*
    ```
2.  Ensure the corresponding Dockerfile exists.

## Adding Artifacts to Your Build Context

1.  Add the artifact path to the `context` key in `buildspec.yml` file.
    ```yaml
    # mxnet/training/buildspec.yml
     20   README.xyz: *<---- Object name (Can be anything)*
     21     source: README-context.rst *<--- Path for the file to be copied*
     22     target: README.rst *<--- Name for the object in** the build context*
    ```
2.  Use `training_context` or `inference_context` for specific image types.
3.  Use the `context` key within a specific image definition for image-specific artifacts.

## Adding a Package

1.  Modify the Dockerfile to include the package installation.
    ```dockerfile
    # mxnet/training/docker/1.6.0/py3/Dockerfile.gpu
    161     octopush
    ```
2.  Rebuild the container.

## Running Tests Locally

Test your changes and ensure image functionality by running local tests.

1.  **Prerequisites:**
    *   EC2 instance or local machine with the repo cloned.
    *   Images you want to test (likely pulled from ECR).
    *   Install testing requirements.
        ```shell
        pip install -r src/requirements.txt
        pip install -r test/requirements.txt
        ```
2.  **Set Environment Variables:**
    *   `DLC_IMAGES`: Space-separated list of ECR URIs to test.
    *   `PYTHONPATH`: Absolute path to the `src/` folder.
    *   `CODEBUILD_RESOLVED_SOURCE_VERSION`: Unique identifier.
    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Change Directory:**
    ```shell
    cd test/dlc_tests
    ```
4.  **Run All Tests (Series):**
    ```shell
    # EC2
    pytest -s -rA ec2/ -n=auto
    # ECS
    pytest -s -rA ecs/ -n=auto
    #EKS
    cd ../
    export TEST_TYPE=eks
    python test/testrunner.py
    ```
    To run tests sequentially remove `-n=auto`.
5.  **Run a specific test file:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```
6.  **Run a specific test function:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```
7.  **SageMaker local mode tests:**
    *   Clone your branch.
    *   Login to ECR
    *   Change to the appropiate directory based on your framework and job type of the image being tested.
    *   Install requirements:
        ```shell
        cd test/sagemaker_tests/mxnet/training/
        pip3 install -r requirements.txt
        ```
    *   Run the pytest command below:
        ```shell
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
         --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
         --py-version 3
        ```
    *   For tensorflow_inference py3 images, run the command below:
       ```shell
       python3 -m  pytest -v integration/local \
       --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference \
       --tag 1.15.2-cpu-py36-ubuntu16.04 --framework-version 1.15.2 --processor cpu
       ```
8.  **SageMaker remote tests:**
    *   Create an IAM role with the `AmazonSageMakerFullAccess` policy.
    *   Change to the correct directory and install requirements (as in Step 7).
    *   Run pytest command below:
       ```shell
       pytest integration/sagemaker/test_mnist.py \
       --region us-west-2 --docker-base-name mxnet-training \
       --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
       --instance-type ml.g5.12xlarge
       ```
    *   For tensorflow_inference py3 images run the below command:
      ```shell
      python3 -m pytest test/integration/sagemaker/test_tfs. --registry {aws_account_id} \
      --region us-west-2  --repo tensorflow-inference --instance-types ml.c5.18xlarge \
      --tag 1.15.2-py3-cpu-build --versions 1.15.2
      ```
9.  **SageMaker benchmark tests:**
    *   Create `sm_benchmark_env_settings.config`.
    *   Add the following (modify as needed):
        ```shell
        export DLC_IMAGES="<image_uri_1-you-want-to-benchmark-test>"
        export BUILD_CONTEXT=PR
        export TEST_TYPE=benchmark-sagemaker
        export CODEBUILD_RESOLVED_SOURCE_VERSION=$USER
        export REGION=us-west-2
        ```
    *   Run:
        ```shell
        source sm_benchmark_env_settings.config
        ```
    *   Test all images
        ```shell
        pip install -r requirements.txt
        python test/testrunner.py
        ```
    *   Test one individual framework image type, run:
        ```shell
        # Assuming that the cwd is deep-learning-containers/
        cd test/dlc_tests
        pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py