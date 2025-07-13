# openpilot: Open Source Driving Automation

**Take control of your driving experience with openpilot, an open-source driving automation system that enhances the driver assistance features in over 300 supported car models.** Explore the full capabilities of openpilot on the [original GitHub repository](https://github.com/commaai/openpilot).

## Key Features

*   **Enhanced Driver Assistance:** Upgrade your car's existing driver assistance systems with advanced features.
*   **Open Source:** Benefit from community contributions and transparency in development.
*   **Wide Car Support:** Compatible with a growing list of over 300+ supported car models.
*   **Continuous Development:** Stay up-to-date with frequent updates and improvements.
*   **Community Driven:** Join a vibrant community of users and developers.

## Getting Started

To use openpilot, you'll need a few things:

1.  **Supported Device:**  A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL: `openpilot.comma.ai` (Setup procedure for the comma 3/3X allows users to enter a URL for custom software.)
3.  **Supported Car:** Verify that your car is on the list of [275+ supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your comma 3/3X to your car.

Detailed instructions for [installing the harness and device](https://comma.ai/setup) are available. Explore running openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/), though it's not plug-and-play.

## Branches

Choose the branch that fits your needs:

*   `release3`:  Stable release branch (`openpilot.comma.ai`).
*   `release3-staging`: Staging branch for early access to new releases (`openpilot-test.comma.ai`).
*   `nightly`: Bleeding-edge development branch; expect instability (`openpilot-nightly.comma.ai`).
*   `nightly-dev`:  Nightly with experimental features for some cars (`installer.comma.ai/commaai/nightly-dev`).
*   `secretgoodopenpilot`: Preview branch with early driving models (`installer.comma.ai/commaai/secretgoodopenpilot`).

## Contributing

openpilot thrives on contributions from users and developers like you.

*   Join the [community Discord](https://discord.comma.ai)
*   Review the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Find code documentation at https://docs.comma.ai
*   Find information on the [community wiki](https://github.com/commaai/openpilot/wiki)

Consider joining the [comma](https://comma.ai/) team, which offers [bounties](https://comma.ai/bounties) for external contributors.

## Safety and Testing

openpilot prioritizes safety with these measures:

*   Adherence to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   Safety model code in `panda` written in C, see [code rigor](https://github.com/commaai/panda#code-rigor).
*   Software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety) in `panda`.
*   Internal hardware-in-the-loop Jenkins test suite.
*   Additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile) in `panda`.
*   Continuous testing on 10 comma devices.

## License

[MIT License](LICENSE)