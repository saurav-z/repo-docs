# openpilot: The Open Source Driver Assistance System

**Upgrade your car's driver assistance with openpilot, an open-source operating system for robotics that enhances the driving experience in over 300 supported vehicles.** ([Original Repository](https://github.com/commaai/openpilot))

## Key Features

*   **Enhanced Driver Assistance:** Improves existing driver-assistance features like Adaptive Cruise Control (ACC) and Lane Keeping Assist (LKA).
*   **Wide Vehicle Compatibility:** Supports 300+ car models, with new additions constantly being made.
*   **Open Source:** Built by comma.ai and a vibrant community.
*   **Continuous Development:** Actively updated with new features and improvements.
*   **Data-Driven:** Utilizes user data (with user consent) to improve and refine autonomous driving models.

## Getting Started

### What You Need

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Use the URL `openpilot.comma.ai` during installation to install the release version.
3.  **Supported Car:** Ensure your vehicle is on [the supported cars list](docs/CARS.md).
4.  **Car Harness:** You'll need a [car harness](https://comma.ai/shop/car-harness) to connect your device.

### Installation

1.  Follow detailed instructions for [how to install the harness and device in a car](https://comma.ai/setup).
2.  Alternative hardware options can be explored on the [openpilot blog](https://blog.comma.ai/self-driving-car-for-free/).

### Quick Start

Run the following command in your terminal:

```bash
bash <(curl -fsSL openpilot.comma.ai)
```

### Available Branches

*   `release3`: ( `openpilot.comma.ai` ) The official release branch.
*   `release3-staging`: ( `openpilot-test.comma.ai` ) Pre-release staging branch.
*   `nightly`: ( `openpilot-nightly.comma.ai` ) Bleeding-edge development branch (may be unstable).
*   `nightly-dev`: ( `installer.comma.ai/commaai/nightly-dev` ) Nightly branch with experimental features for some cars.
*   `secretgoodopenpilot`: ( `installer.comma.ai/commaai/secretgoodopenpilot` ) Preview branch from the autonomy team.

## Contributing

openpilot thrives on community contributions.

*   Join the [community Discord](https://discord.comma.ai)
*   Review the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Find code documentation at https://docs.comma.ai
*   Check the [community wiki](https://github.com/commaai/openpilot/wiki)

## Safety and Testing

*   openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines; see [SAFETY.md](docs/SAFETY.md).
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run on every commit.
*   The safety model is enforced within panda, written in C, see [code rigor](https://github.com/commaai/panda#code-rigor).
*   Panda includes software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Hardware-in-the-loop Jenkins test suite is in place.
*   Panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing using multiple comma devices in a testing environment.

## Resources

*   [Docs](https://docs.comma.ai)
*   [Roadmap](https://docs.comma.ai/contributing/roadmap/)
*   [Contribute](https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md)
*   [Community Discord](https://discord.comma.ai)
*   [Shop](https://comma.ai/shop)

---

<details>
<summary>MIT License</summary>

[Details from original README]
</details>

<details>
<summary>User Data and comma Account</summary>

[Details from original README]
</details>