# openpilot: Drive Smarter, Safer, and Smarter

[<img src="https://github.com/commaai/openpilot/blob/master/docs/assets/openpilot_logo.png?raw=true" alt="openpilot logo" width="200"/>](https://github.com/commaai/openpilot)

**openpilot**, developed by [comma.ai](https://comma.ai), is an open-source, advanced driver-assistance system (ADAS) that enhances the capabilities of your car's existing driver-assist features, currently supporting over 300+ vehicle makes and models. Visit the [original repository](https://github.com/commaai/openpilot) for more information.

## Key Features

*   **Enhanced Driver Assistance:** Upgrade your car's existing features with lane-keeping assist, adaptive cruise control, and more.
*   **Open Source & Community-Driven:** Benefit from continuous improvements and contributions from a vibrant community of developers and users.
*   **Wide Vehicle Support:** Compatible with a growing list of over 300 supported cars.
*   **Easy Installation:** Simple setup process using a comma 3/3X and compatible car harness.
*   **Data-Driven Improvement:** Contribute to the ongoing development of openpilot through data collection and model training (with optional opt-out).

## Getting Started

To begin using openpilot, you'll need the following:

1.  **Compatible Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version by entering the URL `openpilot.comma.ai` during setup.
3.  **Supported Car:** Ensure your vehicle is on the [supported cars list](docs/CARS.md).
4.  **Car Harness:** Purchase a compatible [car harness](https://comma.ai/shop/car-harness) for your vehicle.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

### Branches

Choose the branch that best suits your needs:

| Branch              | URL                              | Description                                                                 |
| ------------------- | -------------------------------- | --------------------------------------------------------------------------- |
| `release3`          | openpilot.comma.ai                | Stable release branch.                                                     |
| `release3-staging`  | openpilot-test.comma.ai           | Staging branch for early access to new releases.                           |
| `nightly`           | openpilot-nightly.comma.ai        | Bleeding-edge development branch; may be unstable.                           |
| `nightly-dev`       | installer.comma.ai/commaai/nightly-dev | Includes experimental development features for some cars.                  |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team with new driving models. |

## Contributing & Community

openpilot thrives on community contributions. We encourage you to:

*   Join the [community Discord](https://discord.comma.ai).
*   Review the [contributing docs](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Access code documentation at https://docs.comma.ai.
*   Find information about running openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki).

[comma.ai](https://comma.ai/) is hiring! Explore open positions and bounties at [comma.ai/jobs#open-positions](https://comma.ai/jobs#open-positions) and [comma.ai/bounties](https://comma.ai/bounties).

## Safety and Testing

openpilot prioritizes safety through:

*   Adherence to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   Rigorous code in panda (written in C) enforcing safety models (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   Software-in-the-loop safety [tests](https://github.com/commaai/panda/tree/master/tests/safety) in panda.
*   Hardware-in-the-loop Jenkins test suite for continuous building and testing.
*   Continuous testing with multiple comma devices replaying routes.

## License

openpilot is released under the [MIT license](LICENSE).

**Disclaimer:**

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT. YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS. NO WARRANTY EXPRESSED OR IMPLIED.**

## User Data and Privacy

By using openpilot, you agree to our [Privacy Policy](https://comma.ai/privacy). Data is used to improve openpilot; data collection can be disabled.

## Resources

*   [Docs](https://docs.comma.ai)
*   [Roadmap](https://docs.comma.ai/contributing/roadmap/)
*   [Contribute](https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md)
*   [Community](https://discord.comma.ai)
*   [Shop](https://comma.ai/shop)
*   [X Follow](https://x.com/comma_ai)
*   [MIT License](LICENSE)