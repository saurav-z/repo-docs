# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started with pre-built, optimized Docker images for training and serving your machine learning models on AWS with the AWS Deep Learning Containers.** ([Original Repo](https://github.com/aws/deep-learning-containers))

## Key Features:

*   **Pre-built and Optimized:**  Ready-to-use Docker images with pre-installed and optimized libraries for TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **GPU and CPU Support:**  Includes NVIDIA CUDA libraries for GPU instances and Intel MKL for CPU instances.
*   **Amazon ECR Availability:**  Images are readily available in Amazon Elastic Container Registry (Amazon ECR) for easy deployment.
*   **Seamless Integration:**  Designed for use with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Building Your Image](#building-your-image)
*   [Upgrading the Framework Version](#upgrading-the-framework-version)
*   [Adding Artifacts to Your Build Context](#adding-artifacts-to-your-build-context)
*   [Adding a Package](#adding-a-package)
*   [Running Tests Locally](#running-tests-locally)

### Getting Started

This section describes how to set up your environment to build and test the Deep Learning Containers (DLCs) on Amazon SageMaker, EC2, ECS, and EKS.

**Example: Building an MXNet GPU Python3 Training Container**

1.  **Prerequisites:**
    *   Access to an AWS account. Configure your environment using the AWS CLI. Recommended: Use an IAM role with the following managed policies:
        *   AmazonEC2ContainerRegistryFullAccess
        *   AmazonEC2FullAccess
        *   AmazonEKSClusterPolicy
        *   AmazonEKSServicePolicy
        *   AmazonEKSServiceRolePolicy
        *   AWSServiceRoleForAmazonEKSNodegroup
        *   AmazonSageMakerFullAccess
        *   AmazonS3FullAccess
    *   Create an ECR repository (e.g., "beta-mxnet-training") in your desired region (e.g., us-west-2).
    *   Docker client set up on your system.

2.  **Setup:**
    *   Clone the repository.
    *   Set environment variables:
        ```shell
        export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
        export REGION=us-west-2
        export REPOSITORY_NAME=beta-mxnet-training
        ```
    *   Log in to ECR:
        ```shell
        aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
        ```
    *   Create a virtual environment and install requirements:
        ```shell
        python3 -m venv dlc
        source dlc/bin/activate
        pip install -r src/requirements.txt
        ```
    *   Perform initial setup:
        ```shell
        bash src/setup.sh mxnet
        ```

### Building Your Image

The Dockerfiles are organized in a specific pattern: `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`.  Build specifications are defined in `mxnet/training/buildspec.yml` files.

1.  **Build all images locally:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  **Build a single image:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet --image_types training --device_types cpu --py_versions py3
    ```
3.  **Arguments:**
    *   `--image_types`: `<training/inference>`
    *   `--device_types`: `<cpu/gpu>`
    *   `--py_versions`: `<py2/py3>` (comma-separated lists for multiple values)

    Example:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet --image_types training --device_types gpu --py_versions py3
    ```

### Upgrading the Framework Version

1.  Modify the `buildspec.yml` file to reflect the new framework version (e.g., changing `version: 1.6.0` to `version: 1.7.0`).
    ```yaml
    # mxnet/training/buildspec.yml
      1   account_id: &ACCOUNT_ID <set-$ACCOUNT_ID-in-environment>
      2   region: &REGION <set-$REGION-in-environment>
      3   framework: &FRAMEWORK mxnet
      4   version: &VERSION 1.6.0 *<--- Change this to 1.7.0*
          ................
    ```
2.  Ensure the Dockerfile for the new version exists (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  Build the container as described above.

### Adding Artifacts to Your Build Context

1.  To copy artifacts into the build context, add them to the buildspec file using the `context` key.
    ```yaml
    # mxnet/training/buildspec.yml
     19 context:
     20   README.xyz: *<---- Object name (Can be anything)*
     21     source: README-context.rst *<--- Path for the file to be copied*
     22     target: README.rst *<--- Name for the object in** the build context*
    ```
2.  Use `training_context` or `inference_context` for image-specific contexts, or add it under the context key for that particular image.

### Adding a Package

1.  Modify the Dockerfile to include the package installation.
    ```dockerfile
    139 RUN ${PIP} install --no-cache --upgrade \
    140     keras-mxnet==2.2.4.2 \
    ...........................
    160     awscli \
    161     octopush  # Add your package
    ```
2.  Build the container as described above. For more details, see [Building AWS Deep Learning Containers Custom Images](custom_images.md).

### Running Tests Locally

Run tests locally to avoid using resources and speed up the development process.

1.  **Prerequisites:**
    *   Ensure images you want to test are available locally (pull from ECR if needed).
    *   Install test requirements:
        ```shell
        cd deep-learning-containers/
        pip install -r src/requirements.txt
        pip install -r test/requirements.txt
        ```
2.  **Environment Variables:**
    *   Set `DLC_IMAGES` (space-separated ECR URIs).
    *   Set `CODEBUILD_RESOLVED_SOURCE_VERSION` to a unique identifier.
    *   Set `PYTHONPATH` to the absolute path of the `src/` folder.

    Example:
    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Run Tests:**
    *   Change directory to `test/dlc_tests`.
    *   Run tests:
        *   For EC2:  `pytest -s -rA ec2/ -n=auto`
        *   For ECS:  `pytest -s -rA ecs/ -n=auto`
        *   For EKS:
            ```shell
            cd ../
            export TEST_TYPE=eks
            python test/testrunner.py
            ```
        *   To run a specific test file: `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py`
        *   To run a specific test function: `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu`
        *   SageMaker local mode tests
            *   Launch a cpu or gpu EC2 instance with latest Deep Learning AMI.
            *   Clone your github branch with changes and run the following commands
                ```shell script
                git clone https://github.com/{github_account_id}/deep-learning-containers/
                cd deep-learning-containers && git checkout {branch_name}
                ```
            *   Login into the ECR repo where the new docker images built exist
                ```shell script
                $(aws ecr get-login --no-include-email --registry-ids ${aws_id} --region ${aws_region})
                ```
            *   Change to the appropriate directory (sagemaker_tests/{framework}/{job_type}) based on framework and job type of the image being tested.
                The example below refers to testing mxnet_training images
                ```shell script
                cd test/sagemaker_tests/mxnet/training/
                pip3 install -r requirements.txt
                ```
            *   To run the SageMaker local integration tests (aside from tensorflow_inference), use the pytest command below:
                ```shell script
                python3 -m pytest -v integration/local --region us-west-2 \
                --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
                    --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
                    --py-version 3
                ```
            *   To test tensorflow_inference py3 images, run the command below:
                ```shell script
                python3 -m  pytest -v integration/local \
                --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference \
                --tag 1.15.2-cpu-py36-ubuntu16.04 --framework-version 1.15.2 --processor cpu
                ```
        *   SageMaker remote tests
            *   Create an IAM role with name “SageMakerRole” in the above account and add the below AWS Manged policies
                ```
                AmazonSageMakerFullAccess
                ```
            *   Change to the appropriate directory (sagemaker_tests/{framework}/{job_type}) based on framework and job type of the image being tested."
                The example below refers to testing mxnet_training images
                ```shell script
                cd test/sagemaker_tests/mxnet/training/
                pip3 install -r requirements.txt
                ```
            *   To run the SageMaker remote integration tests (aside from tensorflow_inference), use the pytest command below:
                ```shell script
                pytest integration/sagemaker/test_mnist.py \
                --region us-west-2 --docker-base-name mxnet-training \
                --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
                --instance-type ml.g5.12xlarge
                ```
            *   For tensorflow_inference py3 images run the below command
                ```shell script
                python3 -m pytest test/integration/sagemaker/test_tfs. --registry {aws_account_id} \
                --region us-west-2  --repo tensorflow-inference --instance-types ml.c5.18xlarge \
                --tag 1.15.2-py3-cpu-build --versions 1.15.2
                ```
        *   SageMaker benchmark tests
            *   Create a file named `sm_benchmark_env_settings.config` in the deep-learning-containers/ folder
            *   Add the following to the file (commented lines are optional):
                ```shell script
                export DLC_IMAGES="<image_uri_1-you-want-to-benchmark-test>"
                # export DLC_IMAGES="$DLC_IMAGES <image_uri_2-you-want-to-benchmark-test>"
                # export DLC_IMAGES="$DLC_IMAGES <image_uri_3-you-want-to-benchmark-test>"
                export BUILD_CONTEXT=PR
                export TEST_TYPE=benchmark-sagemaker
                export CODEBUILD_RESOLVED_SOURCE_VERSION=$USER
                export REGION=us-west-2
                ```
            *   Run:
                ```shell script
                source sm_benchmark_env_settings.config
                ```
            *   To test all images for multiple frameworks, run:
                ```shell script
                pip install -r requirements.txt
                python test/testrunner.py
                ```
            *   To test one individual framework image type, run:
                ```shell script
                # Assuming that the cwd is deep-learning-containers/
                cd test/dlc_tests
                pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py
                ```
            *   The scripts and model-resources used in these tests will be located at:
                ```
                deep-learning-containers/test/dlc_tests/benchmark/sagemaker/<framework-name>/<image-type>/resources/
                ```

Note: SageMaker does not support tensorflow_inference py2 images.