# AWS Deep Learning Containers: Optimized Docker Images for Machine Learning

**Get started quickly with pre-built, optimized Docker images for deep learning frameworks, simplifying model training and deployment on AWS.**  [Explore the original repository](https://github.com/aws/deep-learning-containers).

**Key Features:**

*   **Optimized Environments:** Pre-configured with TensorFlow, TensorFlow 2, PyTorch, and MXNet, along with Nvidia CUDA (for GPU instances) and Intel MKL (for CPU instances) libraries.
*   **Framework Support:**  Images available for popular deep learning frameworks, including TensorFlow, PyTorch, and MXNet.
*   **AWS Integration:** Designed for seamless integration with Amazon SageMaker, Amazon EC2, Amazon ECS, and Amazon EKS.
*   **Pre-built and Ready-to-Use:** Quickly deploy models with readily available images in Amazon Elastic Container Registry (Amazon ECR).
*   **Easy Customization:**  Modify and extend images to meet specific project requirements.
*   **Comprehensive Testing:** Includes robust testing for local and remote environments.

## Getting Started

This section provides instructions for building and testing AWS Deep Learning Containers (DLCs) on various platforms: Amazon SageMaker, EC2, ECS, and EKS.

### Prerequisites
*   **AWS Account Access:**  Configure your AWS CLI with access to your AWS account (IAM user or role recommended).  Recommended managed permissions include:
    *   AmazonEC2ContainerRegistryFullAccess
    *   AmazonEC2FullAccess
    *   AmazonEKSClusterPolicy
    *   AmazonEKSServicePolicy
    *   AmazonEKSServiceRolePolicy
    *   AWSServiceRoleForAmazonEKSNodegroup
    *   AmazonSageMakerFullAccess
    *   AmazonS3FullAccess
*   **ECR Repository:** Create an ECR repository (e.g., "beta-mxnet-training" in us-west-2).
*   **Docker:** Ensure Docker is installed and running on your system.

### Building and Testing a MXNet GPU Container
1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd deep-learning-containers
    ```
2.  **Set Environment Variables:** Replace placeholders with your values.
    ```bash
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
3.  **Login to ECR:**
    ```bash
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
4.  **Create Virtual Environment and Install Requirements:**
    ```bash
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
5.  **Initial Setup:**
    ```bash
    bash src/setup.sh mxnet
    ```

### Building your image

1.  **Build Images with Buildspec.yml:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
2.  **Build a Single Image:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
3.  **Image Type, Device Type, and Python Version Arguments:** Customize the build using:
    *   `--image_types`:  `<training/inference>`
    *   `--device_types`: `<cpu/gpu>`
    *   `--py_versions`: `<py2/py3>`
4.  **Example: Build GPU Training Containers:**
    ```bash
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types gpu \
                       --py_versions py3
    ```

### Upgrading the framework version

1.  **Modify `buildspec.yml`:**  Update the `version` in the buildspec.yml file (e.g., from 1.6.0 to 1.7.0).
2.  **Ensure Dockerfile Exists:** The Dockerfile should reside in the expected path (e.g., `mxnet/docker/1.7.0/py3/Dockerfile.gpu`).
3.  **Build the Container:** Follow the build instructions above.

### Adding artifacts to your build context

1.  **Add Artifacts:** Include files in the build context using the `context` key in your buildspec file.
    ```yaml
    context:
      README.xyz:
        source: README-context.rst
        target: README.rst
    ```
2.  **Targeted Contexts:** Use `training_context` or `inference_context` for image-specific context.
3.  **Specific Image Context:**  Add to the `context` key within a specific image definition.

### Adding a package

1.  **Modify Dockerfile:**  Add the package installation command to the Dockerfile.
    ```dockerfile
    RUN ${PIP} install --no-cache --upgrade \
        keras-mxnet==2.2.4.2 \
        awscli \
        octopush
    ```
2.  **Build the Container:** Follow the build instructions.

### Running tests locally

1.  **Install Test Requirements:**
    ```bash
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Set Environment Variables:**  Specify the ECR image URIs, and set  `PYTHONPATH`
    ```bash
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```
3.  **Navigate to Test Directory:**
    ```bash
    cd test/dlc_tests
    ```
4.  **Run Tests:**
    *   **All Tests (EC2):**  `pytest -s -rA ec2/ -n=auto`
    *   **All Tests (ECS):**  `pytest -s -rA ecs/ -n=auto`
    *   **All Tests (EKS):**
        ```bash
        cd ../
        export TEST_TYPE=eks
        python test/testrunner.py
        ```
    *   **Specific Test File:**  `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py`
    *   **Specific Test Function:** `pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu`

5. **SageMaker Local Mode Tests**
    * Clone repo
    * ECR login
    * Change directory
    * Run Pytest
        ```bash
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
        --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
        --py-version 3
        ```
6.  **SageMaker Remote Tests** (Requires IAM Role and Prerequisites)
    *   Create IAM Role
    *   Change Directory
    *   Run Pytest
        ```bash
        pytest integration/sagemaker/test_mnist.py \
        --region us-west-2 --docker-base-name mxnet-training \
        --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
        --instance-type ml.g5.12xlarge
        ```
7.  **SageMaker Benchmark Tests**
    *   Create `sm_benchmark_env_settings.config`
    *   Populate the config with image uri, BUILD_CONTEXT, TEST_TYPE, CODEBUILD_RESOLVED_SOURCE_VERSION, REGION
    *   Run `source sm_benchmark_env_settings.config`
    *   Run all tests `python test/testrunner.py`
    *   Test individual framework image `pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py`

## License

This project is licensed under the Apache-2.0 License.

`smdistributed.dataparallel` and `smdistributed.modelparallel` are released under the [AWS Customer Agreement](https://aws.amazon.com/agreement/).