## AWS Deep Learning Containers: Build, Train, and Deploy Your Machine Learning Models

Get started with ready-to-use Docker images for machine learning, optimized for speed and performance on AWS.  ([Original Repository](https://github.com/aws/deep-learning-containers))

**Key Features:**

*   **Pre-built Images:** Access optimized Docker images for popular deep learning frameworks, including TensorFlow, PyTorch, MXNet, and more.
*   **GPU and CPU Support:** Benefit from optimized environments for both GPU (Nvidia CUDA) and CPU (Intel MKL) instances, ensuring efficient resource utilization.
*   **AWS Integration:** Seamlessly integrates with Amazon SageMaker, EC2, ECS, and EKS for training, inference, and deployment.
*   **Easy Deployment:** Simplify model building and deployment across AWS services, allowing you to focus on model development.
*   **Flexible Testing:** Test your images locally with pytest, or deploy them on your AWS account.

---

### Getting Started

This guide provides steps to build and test AWS Deep Learning Containers (DLCs) using an example, *MXNet GPU python3 training*.

*   **Prerequisites:**
    *   An active AWS account with appropriate permissions. We recommend using an IAM role with permissions like AmazonEC2ContainerRegistryFullAccess, AmazonEC2FullAccess, AmazonEKSClusterPolicy, AmazonEKSServicePolicy, AmazonEKSServiceRolePolicy, AWSServiceRoleForAmazonEKSNodegroup, AmazonSageMakerFullAccess, and AmazonS3FullAccess.
    *   AWS CLI configured to access your account.
    *   Docker client set up on your system.

1.  **Clone the Repository:** Clone the `aws/deep-learning-containers` repository.
2.  **Set Environment Variables:** Configure the following environment variables:
    ```shell
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-mxnet-training
    ```
3.  **Log in to ECR:** Authenticate with the Amazon Elastic Container Registry (ECR):
    ```shell
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ```
4.  **Set Up Virtual Environment & Install Requirements:**
    ```shell
    python3 -m venv dlc
    source dlc/bin/activate
    pip install -r src/requirements.txt
    ```
5.  **Initial Setup:** Run the setup script:
    ```shell
    bash src/setup.sh mxnet
    ```

### Building Your Image

The build process utilizes `buildspec.yml` files within each framework directory. These files define the Dockerfile paths.

1.  **Build All Images (using buildspec.yml):**
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml --framework mxnet
    ```
    This command builds all images specified in the `buildspec.yml` file.  The first run may take longer as it downloads base layers.

2.  **Build a Single Image:** To build specific images, you can use the following command:
    ```shell
    python src/main.py --buildspec mxnet/training/buildspec.yml \
                       --framework mxnet \
                       --image_types training \
                       --device_types cpu \
                       --py_versions py3
    ```
    Where `--image_types`, `--device_types`, and `--py_versions` are comma separated lists.

3.  **Available arguments:**
    ```shell
    --image_types <training/inference>
    --device_types <cpu/gpu>
    --py_versions <py2/py3>
    ```

### Upgrading the Framework Version

1.  **Modify `buildspec.yml`:** Update the framework version within the corresponding `buildspec.yml` file. For example, to upgrade MXNet to 1.7.0, change the `version` key:
    ```yaml
    # mxnet/training/buildspec.yml
    version: &VERSION 1.7.0
    ```

2.  **Dockerfile:** The dockerfile path follows this pattern:  `mxnet/docker/<version>/<py_version>/Dockerfile.<processor>`

3.  **Build the container:** Build the updated container using the build commands above.

### Adding Artifacts to Your Build Context

To include files in your build context:

1.  **Update `buildspec.yml`:** Add the artifact information under the `context` key in your `buildspec.yml` file. For example:
    ```yaml
    context:
      README.xyz:
        source: README-context.rst
        target: README.rst
    ```

2.  **Specific Image Contexts:** Use `training_context` or `inference_context` to include artifacts only in training or inference images.

3.  **Individual Container Context:** Include the `context` key directly within an image definition for specific container needs.

4.  **Build the Container:** Build the image.

### Adding a Package

1.  **Modify the Dockerfile:** Add the package installation command using `pip install` or similar, within the Dockerfile:
    ```dockerfile
    RUN ${PIP} install --no-cache --upgrade \
        awscli \
        octopush
    ```

2.  **Build the Container:** Rebuild your container to include the new package.

### Running Tests Locally

Run tests locally to ensure image builds and updates:

1.  **Prerequisites**
    *   Make sure you have the images you want to test locally (likely need to pull them from ECR)
    *   Install test requirements
    ```shell
    cd deep-learning-containers/
    pip install -r src/requirements.txt
    pip install -r test/requirements.txt
    ```
2.  **Set Environment Variables:**  Set environment variables, including the image URIs to test:
    ```shell
    export DLC_IMAGES="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3 $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3"
    export PYTHONPATH=$(pwd)/src
    export CODEBUILD_RESOLVED_SOURCE_VERSION="my-unique-test"
    ```

3.  **Change Directory:** Navigate to the test directory:
    ```shell
    cd test/dlc_tests
    ```

4.  **Run Tests (Series):**
    *   **EC2:**  `pytest -s -rA ec2/ -n=auto`
    *   **ECS:**  `pytest -s -rA ecs/ -n=auto`
    *   **EKS:**  `cd ../` then `export TEST_TYPE=eks` followed by `python test/testrunner.py`

    Remove `-n=auto` to run tests sequentially.

5.  **Run a Specific Test File:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py
    ```

6.  **Run a Specific Test Function:**
    ```shell
    pytest -s ecs/mxnet/training/test_ecs_mxnet_training.py::test_ecs_mxnet_training_dgl_cpu
    ```

7.  **SageMaker Local Mode Tests:**
    *   Launch a CPU or GPU EC2 instance with the latest Deep Learning AMI.
    *   Clone your github branch.
    *   Login into the ECR repo where the new docker images built exist.
    *   cd test/sagemaker_tests/{framework}/{job_type}.
    *   pip3 install -r requirements.txt
    *   Run tests, for example:
        ```shell
        python3 -m pytest -v integration/local --region us-west-2 \
        --docker-base-name ${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference \
        --tag 1.6.0-cpu-py36-ubuntu18.04 --framework-version 1.6.0 --processor cpu \
        --py-version 3
        ```

8.  **SageMaker Remote Tests:**
    *   Create an IAM role named "SageMakerRole" with the AmazonSageMakerFullAccess policy.
    *   Follow a structure as per SageMaker Local Mode Tests.
    *   For example:
        ```shell
        pytest integration/sagemaker/test_mnist.py \
        --region us-west-2 --docker-base-name mxnet-training \
        --tag training-gpu-py3-1.6.0 --framework-version 1.6.0 --aws-id {aws_id} \
        --instance-type ml.g5.12xlarge
        ```

9.  **SageMaker Benchmark Tests:**
    *   Create a file named `sm_benchmark_env_settings.config` in the project root.
    *   Populate it with environment variables (see original README for details).
    *   Run `source sm_benchmark_env_settings.config`
    *   Run: `pip install -r requirements.txt` then `python test/testrunner.py`
    *   Or run tests for individual frameworks in test/dlc_tests.
    *   Scripts and model resources will be located at: `deep-learning-containers/test/dlc_tests/benchmark/sagemaker/<framework-name>/<image-type>/resources/`

**Note:**  SageMaker does not support tensorflow_inference py2 images.