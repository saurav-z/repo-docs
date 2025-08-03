# openpilot: Open Source Driver Assistance System

**Upgrade your car's driving capabilities with openpilot, an open-source driver assistance system that enhances the features of 300+ supported car models.**  Check out the original project on [GitHub](https://github.com/commaai/openpilot).

## Key Features:

*   **Advanced Driver Assistance:** Enhance existing driver assistance systems with features like lane keeping, adaptive cruise control, and more.
*   **Wide Compatibility:** Works with over 300 car models, with the list expanding constantly.
*   **Open Source:** Benefit from community contributions, transparency, and the ability to customize the software.
*   **Easy Installation:** Quickly get started by following simple installation steps using a comma 3/3X device and a car harness.
*   **Continuous Improvement:** Data from users is used to train models and improve the system's performance over time.

## Getting Started with openpilot

To use openpilot, you'll need the following:

1.  **Supported Device:** A comma 3/3X device (available at [comma.ai/shop](https://comma.ai/shop/comma-3x)).
2.  **Software:** Install the release version by entering the URL `openpilot.comma.ai` during setup.  You can also choose other branches for testing.
3.  **Supported Car:** Verify that your car model is on the [supported cars list](docs/CARS.md).
4.  **Car Harness:** A compatible [car harness](https://comma.ai/shop/car-harness) to connect your device to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

Select a branch based on your preferences:

*   `release3`: Stable release branch (`openpilot.comma.ai`)
*   `release3-staging`: Staging branch for early access to new releases (`openpilot-test.comma.ai`)
*   `nightly`: Bleeding edge development branch (use with caution - `openpilot-nightly.comma.ai`)
*   `nightly-dev`: Nightly branch with experimental features for certain cars (`installer.comma.ai/commaai/nightly-dev`)
*   `secretgoodopenpilot`: Preview branch with early model merges (`installer.comma.ai/commaai/secretgoodopenpilot`)

## Contributing

openpilot thrives on community involvement.  Here's how you can contribute:

*   Join the [community Discord](https://discord.comma.ai)
*   Review the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Read the [code documentation](https://docs.comma.ai)
*   Visit the [community wiki](https://github.com/commaai/openpilot/wiki)

## Safety and Testing

*   openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.
*   Extensive software-in-the-loop and hardware-in-the-loop tests are performed regularly.
*   The code that enforces the safety model is written in C, ensuring rigorous code standards.

## Licensing

openpilot is released under the [MIT License](LICENSE).

**Disclaimer:** *This is alpha-quality software for research purposes only.  It is not a product.  You are responsible for complying with local laws and regulations.  NO WARRANTY EXPRESSED OR IMPLIED.*

## User Data and Privacy

By default, openpilot uploads driving data to comma.ai servers to improve the software.  You can access your data through [comma connect](https://connect.comma.ai/). You can disable data collection if desired.  By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).