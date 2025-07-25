# openpilot: Open Source Driving Automation for Your Car

**openpilot is an open-source, community-driven driving automation system that upgrades the driver assistance features in over 300 supported car models.** ([Original Repo](https://github.com/commaai/openpilot))

## Key Features:

*   **Advanced Driver Assistance:** Provides lane keeping, adaptive cruise control, and automatic steering.
*   **Wide Car Support:** Compatible with a rapidly growing list of over 300+ supported vehicles.
*   **Community-Driven Development:** Benefit from continuous improvements and contributions from a vibrant open-source community.
*   **Easy Installation:**  Simple setup using a comma 3/3X and a compatible car harness.
*   **Data-Driven Improvement:** Utilizes user data to train and refine driving models, leading to safer and more reliable performance.
*   **Regular Updates:** Benefit from new features and improvements through different branch options.
*   **ISO26262 Guidelines:** The code observes ISO26262 guidelines, see [SAFETY.md](docs/SAFETY.md) for more details.

## How to Get Started:

To use openpilot, you'll need:

1.  **Compatible Device:**  A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the openpilot release version using the URL `openpilot.comma.ai` during the setup of your comma 3/3X.
3.  **Supported Car:** Ensure your vehicle is on the list of [275+ supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your device to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches:

Access different versions of openpilot for various purposes:

*   `release3`: `openpilot.comma.ai` - The stable release branch.
*   `release3-staging`: `openpilot-test.comma.ai` - Get early access to upcoming releases.
*   `nightly`: `openpilot-nightly.comma.ai` - The development branch, offering the latest features (may be unstable).
*   `nightly-dev`: `installer.comma.ai/commaai/nightly-dev` - Includes experimental development features.
*   `secretgoodopenpilot`: `installer.comma.ai/commaai/secretgoodopenpilot` - Preview branch for new driving models.

## Contributing and Community:

openpilot thrives on community involvement. Here's how you can contribute:

*   Join the [community Discord](https://discord.comma.ai).
*   Explore the [contributing docs](docs/CONTRIBUTING.md).
*   Check out the [openpilot tools](tools/).
*   Access code documentation at https://docs.comma.ai.
*   Find information on the [community wiki](https://github.com/commaai/openpilot/wiki).

## Safety and Testing:

*   openpilot follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run with every commit.
*   The code enforcing the safety model is in `panda` and written in C.
*   `panda` has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety) and hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing on multiple devices to ensure reliability.

---

**Disclaimer:**

*   openpilot is provided under the MIT license.
*   **THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT. YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS. NO WARRANTY EXPRESSED OR IMPLIED.**