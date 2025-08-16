# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Accelerate your machine learning workflows with AWS Deep Learning Containers, pre-built and optimized Docker images for popular deep learning frameworks.**  [View the original repository](https://github.com/aws/deep-learning-containers)

## Key Features

*   **Pre-built and Optimized:** Ready-to-use Docker images pre-configured with popular deep learning frameworks like TensorFlow, PyTorch, and MXNet, along with optimized libraries for CPU, GPU, and Intel MKL.
*   **Framework Support:**  Images available for TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Integration with AWS Services:** Seamlessly integrated with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Hardware Acceleration:** Optimized for both CPU and GPU instances, leveraging Nvidia CUDA for GPU acceleration.
*   **Easy Deployment:** Available in Amazon Elastic Container Registry (Amazon ECR) for easy deployment and management.

## Getting Started

This section outlines how to build and test the Deep Learning Containers.

### Prerequisites

*   An active AWS account, configured with the AWS CLI.  We recommend using an IAM role with the following managed permissions:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
*   Docker client set up on your system.

### Example: Building an MXNet GPU Python3 Training Container

1.  **Set up your environment:**

    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```

2.  **Create an ECR repository:**
    *   Create an ECR repository named "beta-mxnet-training" in the us-west-2 region.

3.  **Log in to ECR:**

    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```

4.  **Clone the repository and install requirements:**

    ```shell
    git clone <your_repo_url>  # Replace with the actual repo URL
    cd deep-learning-containers/
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```

5.  **Perform initial setup:**

    ```shell
    bash src/setup.sh mxnet
    ```

## Building Your Image

### Building Dockerfiles Locally

The paths to the Dockerfiles follow a specific pattern: `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`.  These paths are defined in the `buildspec.yml` files (e.g., `mxnet/training/buildspec.yml`).  To build a specific version or introduce a new version, modify the `buildspec.yml` file.

1.  **Build all Dockerfiles:**

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

3.  **Build a specific set of images (example: GPU training containers):**

    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

### Upgrading the Framework Version

1.  **Modify `buildspec.yml`:** Update the `version` key in the relevant `buildspec.yml` file.
    ```yaml
    # mxnet/training/buildspec.yml
      4   version: &VERSION 1.7.0  # Change to the new version
    ```
2.  **Ensure Dockerfile exists:**  The Dockerfile should be at `mxnet/docker/<version>/<python_version>/Dockerfile.<processor>`.
3.  **Build the container:** Use the build commands described above.

### Adding Artifacts to Your Build Context

1.  **Add artifacts to the `context` key:** Add your artifact files to the framework `buildspec.yml` file under the `context` key.
    ```yaml
    # mxnet/training/buildspec.yml
     19 context:
     20   README.xyz:
     21     source: README-context.rst
     22     target: README.rst
    ```

2.  **Use `training_context` or `inference_context`:** For images add context under  `training_context` or `inference_context`.

3. **Add for single container:**  To add for single container add to the `context` key for the specific image.
4.  **Build the container:** Use the build commands.

### Adding a Package

1.  **Modify the Dockerfile:** Add the package install command to the Dockerfile (e.g., using `pip install`).
    ```dockerfile
    # mxnet/training/docker/1.6.0/py3/Dockerfile.gpu
    161     octopush  # Add the package install
    ```
2.  **Build the container:** Use the build commands.

## Running Tests Locally

This section details how to run tests locally.  Ensure you have the necessary images pulled from ECR.

1.  **Install test requirements:**

    ```shell
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```

2.  **Set environment variables:**

    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

3.  **Navigate to the test directory:**

    ```shell
    cd test/dlc_tests
    ```

4.  **Run all tests (in series) for a given platform:**

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

5.  **Run a specific test file:**

    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

6.  **Run a specific test function:**

    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

7.  **Run SageMaker local mode tests:**

    *   Set up an EC2 instance with the latest Deep Learning AMI or use your local machine
    *   Clone your github branch with changes
    *   Login into the ECR repo where the new docker images built exist
    *   Change to the appropriate directory `sagemaker_tests/{framework}/{job_type}`.
    *   Use `pytest` command below
        ```shell
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
         --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
         --py-version 3
        ```

8. **Run SageMaker remote tests:**
    *   Create an IAM role with name “SageMakerRole” in the above account and add the below AWS Manged policies
    ```
    AmazonSageMakerFullAccess
    ```
    *   Change to the appropriate directory `sagemaker_tests/{framework}/{job_type}`.
    *   Use `pytest` command below
        ```shell
        pytest integration/sagemaker/test_mnist.py \
        --region us-west-2 --docker-base-name mxnet-training \
        --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
        --instance-type ml.g5.12xlarge
        ```

9. **Run SageMaker benchmark tests:**

    *   Create `sm_benchmark_env_settings.config`
    *   Add the settings
    *   Run
        ```shell
        source sm_benchmark_env_settings.config
        ```
    *   To test multiple framework images, run `python test/testrunner.py`.
    *   To test one individual framework image type, run
       `pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py`

**Note:**  SageMaker does not support tensorflow_inference py2 images.