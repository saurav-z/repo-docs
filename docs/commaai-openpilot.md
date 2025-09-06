# openpilot: Open Source Driver Assistance for Your Car

**Upgrade your driving experience with openpilot, an open-source operating system that adds advanced driver-assistance features to over 300 supported car models.** ([Original Repo](https://github.com/commaai/openpilot))

## Key Features

*   **Advanced Driver-Assistance Systems (ADAS):** Enables features like adaptive cruise control, lane keeping assist, and automatic emergency braking.
*   **Wide Car Compatibility:** Supports over 300 car models. Check the [supported cars](docs/CARS.md) list.
*   **Community-Driven Development:** Benefit from a constantly evolving platform, enhanced by a vibrant community of developers and users.
*   **Open Source & Customizable:** Modify and adapt the software to suit your preferences, fostering innovation and community contributions.
*   **Regular Updates:** Stay up-to-date with the latest features and improvements.

## Getting Started

To use openpilot, you'll need:

1.  **Supported Device:** A [comma 3X](https://comma.ai/shop/comma-3x).
2.  **Software:** Install openpilot using the URL `openpilot.comma.ai` in the comma 3X setup.
3.  **Supported Car:** Ensure your vehicle is on the [supported cars](docs/CARS.md) list.
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your comma 3X.

Detailed setup instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

*   `release3`: `openpilot.comma.ai` - The stable release branch.
*   `release3-staging`: `openpilot-test.comma.ai` - Staging branch for early access to new releases.
*   `nightly`: `openpilot-nightly.comma.ai` - Bleeding edge development branch (unstable).
*   `nightly-dev`: `installer.comma.ai/commaai/nightly-dev` - Nightly with experimental features for some cars.

## Contributing

Join the openpilot community and contribute to its development:

*   [Join the Discord Community](https://discord.comma.ai)
*   Review the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Read the [documentation](https://docs.comma.ai)
*   Check out the [community wiki](https://github.com/commaai/openpilot/wiki)

## Safety and Testing

openpilot is designed with safety in mind, adhering to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)). The project employs several safety measures, including:

*   Software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   Code enforcing the safety model is written in C within panda.
*   Software-in-the-loop safety [tests](https://github.com/commaai/panda/tree/master/tests/safety) within panda.
*   Hardware-in-the-loop Jenkins test suite.
*   Continuous testing on devices.

## License

openpilot is released under the [MIT License](LICENSE).

**Disclaimer:** This is alpha quality software for research purposes only. It is not a product, and you are responsible for complying with local laws and regulations. No warranty is expressed or implied.

## Data Privacy

By default, openpilot uploads driving data to our servers. You can access your data through [comma connect](https://connect.comma.ai/). See the [Privacy Policy](https://comma.ai/privacy) for more details. You are free to disable data collection.