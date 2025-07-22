# Openpilot: Upgrade Your Car's Driver Assistance System with Open Source Technology

**Openpilot is a cutting-edge open-source software that enhances your car's driver assistance features, supporting over 300+ vehicles.**

[Check out the original repo](https://github.com/commaai/openpilot)

## Key Features:

*   **Open Source:** Benefit from a community-driven project with transparent code and continuous improvements.
*   **Wide Vehicle Support:** Compatible with over 300+ supported cars.
*   **Advanced Driver Assistance:** Upgrades existing driver assistance systems with features like lane keeping, adaptive cruise control, and more.
*   **Regular Updates:** Stay up-to-date with the latest features and improvements through frequent releases.
*   **Community Driven:** Join a vibrant community of developers and users for support, collaboration, and innovation.

## Getting Started:

### What You'll Need:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai` during setup.
3.  **Supported Car:** Ensure your car is on [the supported car list](docs/CARS.md).
4.  **Car Harness:** You'll need a [car harness](https://comma.ai/shop/car-harness) to connect your comma 3/3X to your car.

### Installation:

Detailed instructions for installation can be found at [comma.ai/setup](https://comma.ai/setup).

### Software Branches:

*   **`release3`:** Stable release branch (`openpilot.comma.ai`)
*   **`release3-staging`:** Staging branch for early access to new releases (`openpilot-test.comma.ai`)
*   **`nightly`:** Bleeding-edge development branch (unstable) (`openpilot-nightly.comma.ai`)
*   **`nightly-dev`:** Nightly with experimental features for some cars (`installer.comma.ai/commaai/nightly-dev`)
*   **`secretgoodopenpilot`:** Preview branch for new driving models (`installer.comma.ai/commaai/secretgoodopenpilot`)

## Contributing:

Openpilot thrives on community contributions!

*   Join the [Discord community](https://discord.comma.ai) to connect with other users and developers.
*   Contribute to the project by checking out the [contributing docs](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Refer to the [code documentation](https://docs.comma.ai) and the [community wiki](https://github.com/commaai/openpilot/wiki) for more information.
*   [Comma is hiring](https://comma.ai/jobs#open-positions) and offers bounties.

## Safety and Testing:

*   Openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines. See [SAFETY.md](docs/SAFETY.md) for details.
*   Extensive software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) are run on every commit.
*   Safety-critical code is written in C within the [panda](https://github.com/commaai/panda#code-rigor) project.
*   [Safety tests](https://github.com/commaai/panda/tree/master/tests/safety) are in place for the panda project.
*   Hardware-in-the-loop testing is performed internally.
*   Continuous testing is done on 10 comma devices.

## License

Openpilot is licensed under the [MIT license](LICENSE).

**Disclaimer:** This is alpha-quality software for research purposes only. You are responsible for complying with local laws and regulations. No warranty is expressed or implied.

## User Data and Comma Account

By default, openpilot uploads driving data to our servers to improve the software. Users can access their data through [comma connect](https://connect.comma.ai/). The user is free to disable data collection.  By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).