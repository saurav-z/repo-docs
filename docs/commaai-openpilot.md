# openpilot: Open Source Driver Assistance System

**Upgrade your car's driver assistance system with openpilot, a cutting-edge open-source project.**

[Visit the original repo on GitHub](https://github.com/commaai/openpilot)

## Key Features

*   **Advanced Driver Assistance:** Enhance your driving experience with features like adaptive cruise control, lane keeping assist, and automatic lane changes.
*   **Wide Vehicle Compatibility:** Supports 300+ car models, with new vehicles continually being added.
*   **Open Source & Community Driven:** Benefit from a collaborative development model with contributions from the open-source community.
*   **Easy Installation:** Install openpilot on a comma 3X device (available at [comma.ai/shop](https://comma.ai/shop/comma-3x)) using a simple URL.
*   **Continuous Updates:** Stay up-to-date with the latest features and improvements through the release, staging, and nightly branches.
*   **Safety Focused:** Built with safety in mind, adhering to ISO26262 guidelines and undergoing rigorous testing.

## How to Get Started

### Required Components

1.  **Supported Device:** A comma 3X device from [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the latest release version by entering the URL `openpilot.comma.ai` during setup on your comma 3X device.
3.  **Supported Car:**  Ensure your vehicle is compatible by checking the list of supported cars [here](docs/CARS.md).
4.  **Car Harness:** Purchase a car harness compatible with your vehicle from [comma.ai/shop/car-harness](https://comma.ai/shop/car-harness).

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

### Branches

Choose the branch that fits your needs:

| Branch            | URL                     | Description                                                                          |
|-------------------|-------------------------|--------------------------------------------------------------------------------------|
| `release3`        | openpilot.comma.ai       | The stable, release branch.                                                           |
| `release3-staging`| openpilot-test.comma.ai | Get new releases slightly earlier.                                                |
| `nightly`         | openpilot-nightly.comma.ai| Bleeding-edge development branch; expect instability.                                |
| `nightly-dev`     | installer.comma.ai/commaai/nightly-dev | Experimental development features for certain cars.                               |

## Contribute to openpilot

Join the community and help improve openpilot!

*   [Join the Community Discord](https://discord.comma.ai)
*   [Contribute to the Project](docs/CONTRIBUTING.md)
*   Explore [openpilot tools](tools/)
*   Find code documentation at [https://docs.comma.ai](https://docs.comma.ai)
*   Learn more on the [community wiki](https://github.com/commaai/openpilot/wiki)
*   [Check for open positions and bounties](https://comma.ai/jobs#open-positions) if you want to work on openpilot.

## Safety and Testing

openpilot prioritizes safety with:

*   Adherence to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines ([SAFETY.md](docs/SAFETY.md)).
*   Software-in-the-loop tests run on every commit ([.github/workflows/selfdrive_tests.yaml](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)).
*   Rigorous code rigor in the panda project ([code rigor](https://github.com/commaai/panda#code-rigor)).
*   Hardware-in-the-loop and software-in-the-loop safety tests in panda ([tests](https://github.com/commaai/panda/tree/master/tests/safety)).
*   Continuous testing with hardware-in-the-loop Jenkins test suite.
*   Continuous testing in a testing closet containing 10 comma devices.

## License and Data Usage

*   openpilot is released under the [MIT License](LICENSE).
*   openpilot uploads driving data to our servers to improve the system; you can disable this in the settings.
*   By using openpilot, you agree to the [Privacy Policy](https://comma.ai/privacy).