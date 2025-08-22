# AWS Deep Learning Containers: Pre-built Docker Images for ML

**Get up and running quickly with pre-configured Docker images for training and serving your machine learning models on AWS.**  [View the original repository](https://github.com/aws/deep-learning-containers).

## Key Features

*   **Optimized Environments:** Pre-configured with popular deep learning frameworks like TensorFlow, PyTorch, and MXNet, as well as essential libraries like NVIDIA CUDA (for GPU instances) and Intel MKL (for CPU instances).
*   **Amazon ECR Availability:** Easily access and deploy DLCs from the Amazon Elastic Container Registry (ECR).
*   **SageMaker Integration:** Designed for seamless use within Amazon SageMaker for training, inference, and transformation jobs.
*   **Broad Platform Support:** Tested and compatible with Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Customization:**  Easily build custom images by modifying existing Dockerfiles or adding your own packages.
*   **Comprehensive Testing:**  Includes pytest-based testing for EC2, ECS, EKS, and SageMaker environments.

## Getting Started

This section outlines the steps to build and test DLCs, using the example of building an MXNet GPU Python 3 training container.

**Prerequisites:**

*   An AWS account with configured AWS CLI access (IAM user or role). Recommended IAM permissions are listed in the original README.
*   An ECR repository (e.g., "beta-mxnet-training") created in your desired region.
*   Docker installed and configured on your system.

**Steps:**

1.  **Clone the Repository and Set Environment Variables:**
    ```shell
    git clone <repository_url>
    cd deep-learning-containers
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
2.  **Log in to ECR:**
    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
3.  **Create and Activate a Virtual Environment & Install Requirements:**
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
4.  **Perform Initial Setup:**
    ```shell
    bash src/setup.sh mxnet
    ```

## Building Your Image

The structure of the Dockerfiles follows a consistent pattern, specified by the `buildspec.yml` files within the framework directories.  Modify the `buildspec.yml` to specify versions or create new builds.

1.  **Build All Images (as defined in buildspec.yml):**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```

2.  **Build a Single Image:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```

3.  **Image Type, Device Type, and Python Version Arguments:**

    *   `--image_types`:  `training` or `inference`
    *   `--device_types`:  `cpu` or `gpu`
    *   `--py_versions`:  `py2` or `py3`

4.  **Example: Build All GPU Training Containers:**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

## Upgrading the Framework Version

1.  **Modify `buildspec.yml`:**  Change the `version` key to the desired framework version (e.g., from 1.6.0 to 1.7.0).
    ```yaml
    # mxnet/training/buildspec.yml
      4   version: &VERSION 1.7.0
    ```
2.  **Ensure Dockerfile Exists:** Verify the Dockerfile for the new version exists (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  **Build the Container:**  Use the build commands as described above.

## Adding Artifacts to Your Build Context

Add files to the build context using the `context`, `training_context`, or `inference_context` keys within the `buildspec.yml` file.

1.  **Add an Artifact to All Images:**
    ```yaml
     19 context:
     20   README.xyz: *<---- Object name (Can be anything)*
     21     source: README-context.rst *<--- Path for the file to be copied*
     22     target: README.rst *<--- Name for the object in** the build context*
    ```
2.  **Add an Artifact to Training or Inference Images:**
    ```yaml
     23   training_context: &TRAINING_CONTEXT
     24     README.xyz:
     25       source: README-context.rst
     26       target: README.rst
    ```
3.  **Add an Artifact to a Specific Image:**
    ```yaml
     50       context:
     51         <<: *TRAINING_CONTEXT
     52         README.xyz:
     53           source: README-context.rst
     54           target: README.rst
    ```
4.  **Build the Container:** Use the build commands as described above.

## Adding a Package

1.  **Modify the Dockerfile:** Add the package installation command to the `Dockerfile`.
    ```dockerfile
    160     awscli \
    161     octopush
    ```
2.  **Build the Container:** Use the build commands as described above.

## Running Tests Locally

Test your changes locally using pytest to validate images before deployment.

1.  **Prerequisites:** Ensure the images you want to test are accessible (likely pulled from ECR).  Install testing requirements.
    ```shell
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Set Environment Variables:**
    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Navigate to Test Directory:**
    ```shell
    cd test/dlc_tests
    ```
4.  **Run All Tests for a Platform (EC2, ECS):**

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

5.  **Run a Specific Test File:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

6.  **Run a Specific Test Function:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

7.  **SageMaker Local Mode Tests:**  Follow the instructions for EC2 instance setup.  Run the pytest command provided in the original README, adjusting the parameters as necessary for your images.
    ```shell
    python3 -m pytest -v integration/local --region us-west-2 \
    --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
     --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
     --py-version 3
    ```

8.  **SageMaker Remote Tests:**  Requires IAM role setup. Run the pytest command with appropriate parameters, as outlined in the original README.

9.  **SageMaker Benchmark Tests:** Create `sm_benchmark_env_settings.config`, configure environment variables, and run `testrunner.py` or specific pytest commands as indicated in the original README.

**Note:** SageMaker does not support tensorflow_inference py2 images.