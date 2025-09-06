# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get pre-built, optimized Docker images for training and deploying your machine learning models on AWS with ease.  [Learn more at the original repo](https://github.com/aws/deep-learning-containers).**

## Key Features

*   **Framework Support:** TensorFlow, TensorFlow 2, PyTorch, and MXNet.
*   **Optimized Environments:** Pre-configured with CUDA, Intel MKL, and other essential libraries for optimal performance.
*   **Amazon ECR Availability:** Easily accessible within the Amazon Elastic Container Registry (Amazon ECR).
*   **SageMaker Integration:** Seamlessly integrates with Amazon SageMaker for training, inference, and more.
*   **Platform Compatibility:** Tested and validated on Amazon EC2, Amazon ECS, and Amazon EKS.

## Getting Started

Follow these steps to build and test AWS Deep Learning Containers:

### Prerequisites
* AWS Account setup with necessary IAM Permissions: [AmazonEC2ContainerRegistryFullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess), [AmazonEC2FullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEC2FullAccess), [AmazonEKSClusterPolicy](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEKSClusterPolicy), [AmazonEKSServicePolicy](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEKSServicePolicy), [AmazonEKSServiceRolePolicy](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEKSServiceRolePolicy), [AWSServiceRoleForAmazonEKSNodegroup](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AWSServiceRoleForAmazonEKSNodegroup), [AmazonSageMakerFullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonSageMakerFullAccess), [AmazonS3FullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonS3FullAccess)
* Docker installed on your system.

### Setup for MXNet GPU Python 3 Training Container (Example)

1.  **AWS Account Setup:** Configure your AWS CLI with an IAM user or role.
2.  **ECR Repository:** Create an ECR repository (e.g., "beta-mxnet-training") in your desired region.
3.  **Clone the Repository:** Clone the `aws/deep-learning-containers` GitHub repository.
4.  **Environment Variables:** Set the following environment variables:

    ```bash
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```

5.  **ECR Login:** Log in to ECR using Docker:

    ```bash
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```

6.  **Virtual Environment and Dependencies:**
    ```bash
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
7.  **Initial Setup:** Run the setup script:

    ```bash
    bash src/setup.sh mxnet
    ```

## Building Your Image

*   The dockerfiles are specified by the buildspec.yml files residing in `<framework>/<training|inference>/buildspec.yml`.
*   To build all images specified in buildspec.yml:
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
*   To build a single image:
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
    Use comma separated lists for `--image_types`, `--device_types` and `--py_versions`.

## Upgrading the Framework Version

1.  Modify the `version` key in the relevant `buildspec.yml` file.
2.  Ensure the corresponding Dockerfile exists in the correct directory structure.
3.  Build the container as described above.

## Adding Artifacts to Your Build Context

1.  Add artifact paths in the `context` section of the `buildspec.yml`.

## Adding a Package

1.  Modify the Dockerfile to include the package installation using `pip`.

## Running Tests Locally

1.  **Prepare your environment:**  Ensure you have the images you want to test locally (e.g., pull from ECR).  Install test requirements.

    ```bash
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```

2.  **Set Environment Variables:**  Export necessary environment variables for testing.

    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Change Directory:** Change your directory to the test directory.

    ```bash
    cd test/dlc_tests
    ```
4.  **Run Tests:** Execute tests using pytest.  Examples:

    *   Run all tests (in series) for a given platform:
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
    *   Run a specific test file:

        ```bash
        pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
        ```
    *   Run a specific test function:

        ```bash
        pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
        ```

5.  **SageMaker Local Mode Tests**:

    *   Launch an EC2 instance with the latest Deep Learning AMI.
    *   Clone the repo and run:

        ```bash
        $(aws ecr get-login --no-include-email --registry-ids ${aws_id} --region ${aws_region})
        cd test/sagemaker_tests/mxnet/training/
        pip3 install -r requirements.txt
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
        --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
        --py-version 3
        ```
   * For tensorflow_inference py3 images, run the command below:
     ```bash
     python3 -m  pytest -v integration/local \
     --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference \
     --tag 1.15.2-cpu-py36-ubuntu16.04 --framework-version 1.15.2 --processor cpu
     ```
6.  **SageMaker Remote Tests**: Follow setup including IAM Role configuration ("SageMakerRole").  Run tests using pytest.  (Example)

    ```bash
    pytest integration/sagemaker/test_mnist.py \
    --region us-west-2 --docker-base-name mxnet-training \
    --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
    --instance-type ml.g5.12xlarge
    ```

    *   For tensorflow_inference py3 images run the below command:
        ```bash
        python3 -m pytest test/integration/sagemaker/test_tfs. --registry {aws_account_id} \
        --region us-west-2  --repo tensorflow-inference --instance-types ml.c5.18xlarge \
        --tag 1.15.2-py3-cpu-build --versions 1.15.2
        ```
7.  **SageMaker Benchmark Tests**:

    *   Create `sm_benchmark_env_settings.config`.
    *   Populate with `DLC_IMAGES`, `BUILD_CONTEXT`, `TEST_TYPE`, `CODEBUILD_RESOLVED_SOURCE_VERSION`, and `REGION`.
    *   Source this file.
    *   Run `python test/testrunner.py`
    *   or run an individual framework test from the `test/dlc_tests` directory.

```
Note: SageMaker does not support tensorflow_inference py2 images.