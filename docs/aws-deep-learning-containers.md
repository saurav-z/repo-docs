# AWS Deep Learning Containers: Ready-to-Use Docker Images for Machine Learning

**Get started quickly with optimized Docker images for training and deploying your machine learning models on AWS using [AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers).**

## Key Features

*   **Optimized Environments:** Pre-configured with popular deep learning frameworks (TensorFlow, PyTorch, MXNet) and libraries like NVIDIA CUDA (for GPU) and Intel MKL (for CPU).
*   **Seamless Integration:** Designed for use with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Pre-built Images:** Available in Amazon Elastic Container Registry (Amazon ECR), ready for immediate use.
*   **Framework Support:**  Includes support for TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Flexible Deployment:** Suitable for a variety of machine learning workloads, from training to inference.

## Getting Started

### Prerequisites

*   An AWS account with the necessary permissions.  Recommended IAM role configuration:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess

*   Docker installed and configured on your system.
*   AWS CLI configured with access to your AWS account.

### Example: Building an MXNet GPU Training Container

1.  **Set up your environment:**
    *   Clone the repository.
    *   Set the following environment variables, replacing with your actual values:
        ```shell
        export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
        export REGION=us-west-2
        export REPOSITORY_NAME=beta-mxnet-training
        ```
2.  **Login to ECR:**
    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
3.  **Create a virtual environment and install dependencies:**
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
4.  **Perform initial setup:**
    ```shell
    bash src/setup.sh mxnet
    ```
5.  **Build the Docker Image:**

    *   To build all images defined in the buildspec.yml:
        ```shell
        python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
        ```
    *   To build a specific image:
        ```shell
        python src/main.py --buildspec mxnet/training/buildspec.yml \
                           --framework mxnet \
                           --image_types training \
                           --device_types cpu \
                           --py_versions py3
        ```

## Building Your Image

The Dockerfile paths follow a specific pattern:  `mxnet/training/docker/<version>/<python_version>/Dockerfile.<processor>`.  These paths are defined in the `buildspec.yml` files located in the respective framework directories (e.g., `mxnet/training/buildspec.yml`).

### Upgrading Framework Versions

To update to a new framework version (e.g., MXNet 1.7.0):

1.  Modify the `buildspec.yml` file to reflect the new version.
2.  Ensure the Dockerfile for the new version exists.
3.  Build the container using the instructions above.

### Adding Artifacts to Your Build Context

To include additional files in your build context:

1.  Add the artifact details (source and target paths) to the `context` section of the relevant `buildspec.yml` file.  You can add it to the global `context`,  `training_context`, `inference_context`, or a specific image's context.
2.  Rebuild the container.

### Adding Packages

1.  Modify the Dockerfile to include the `pip install` command for the desired package.
2.  Rebuild the container.  For details on customizing, see the [Building AWS Deep Learning Containers Custom Images](custom_images.md).

## Running Tests Locally

Before pushing your changes, run tests locally by first pulling the images:
1. Install the requirements for tests.
```shell
cd deep-learning-containers/
pip install -r src/requirements.txt
pip install -r test/requirements.txt
```
2.  **Set up environment variables:**
    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Change directory to the test directory:**
    ```shell
    cd test/dlc_tests
    ```
4.  **Run tests:**
    *   For EC2:
        ```shell
        pytest -s -rA ec2/ -n=auto
        ```
    *   For ECS:
        ```shell
        pytest -s -rA ecs/ -n=auto
        ```
        
    *   For EKS:
        ```shell
        cd ../
        export TEST_TYPE=eks
        python test/testrunner.py
        ```
    *   To run a specific test file:
        ```shell
        pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
        ```
    *   To run a specific test function:
        ```shell
        pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
        ```
    *   For SageMaker local mode tests, launch a cpu or gpu EC2 instance with latest Deep Learning AMI, clone repo, and setup as described in the original README.

    *   For SageMaker remote tests, create an IAM role named `SageMakerRole` and setup the integration tests as described in the original README.

5.  **SageMaker Benchmark tests**: Follow steps outlined in the original README.

## License

This project is licensed under the Apache-2.0 License.  `smdistributed.dataparallel` and `smdistributed.modelparallel` are released under the [AWS Customer Agreement](https://aws.amazon.com/agreement/).

**[Return to the Original Repository](https://github.com/aws/deep-learning-containers)**