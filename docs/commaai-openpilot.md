# openpilot: The Open Source Autonomous Driving System

**Transform your driving experience with openpilot, an open-source autonomous driving system that enhances driver assistance in hundreds of supported vehicles.** Learn more and contribute to the project on the original [openpilot GitHub repository](https://github.com/commaai/openpilot).

### Key Features:

*   **Enhanced Driver Assistance:** Upgrade the driver assistance system in over 300 supported vehicles.
*   **Community-Driven Development:** Benefit from a vibrant community and contribute to the project.
*   **Open Source:**  Fully open-source and free to use.
*   **Regular Updates:** Benefit from continuous improvements and new features.

### Getting Started with openpilot:

To use openpilot, you'll need the following:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai`.
3.  **Supported Car:** One of the [275+ supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your comma 3/3X.

Detailed instructions for installation are available at [comma.ai/setup](https://comma.ai/setup).

### Branches:

Choose the appropriate branch based on your needs:

| Branch            | URL                         | Description                                                                      |
| ----------------- | --------------------------- | -------------------------------------------------------------------------------- |
| `release3`        | openpilot.comma.ai           | The stable release branch.                                                       |
| `release3-staging` | openpilot-test.comma.ai      | Staging branch for early access to new releases.                                |
| `nightly`         | openpilot-nightly.comma.ai    | Bleeding-edge development branch; may be unstable.                             |
| `nightly-dev`     | installer.comma.ai/commaai/nightly-dev  | Nightly with experimental development features for some cars. |

### Contributing:

openpilot thrives on community contributions.  Here's how you can get involved:

*   Join the [community Discord](https://discord.comma.ai)
*   Review the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Access code documentation at https://docs.comma.ai
*   Find more information on the [community wiki](https://github.com/commaai/openpilot/wiki)

### Safety and Testing:

openpilot prioritizes safety with rigorous testing and adherence to safety standards:

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.
*   Features software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   Employs C-based code for the safety model within panda ([code rigor](https://github.com/commaai/panda#code-rigor)).
*   Includes hardware-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety) and additional Jenkins tests.

### License and Data Usage:

*   openpilot is released under the [MIT License](LICENSE).
*   By default, driving data is uploaded to our servers to train better models. Users can disable data collection.
*   Refer to the [Privacy Policy](https://comma.ai/privacy) for details on data usage.