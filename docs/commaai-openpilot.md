# openpilot: Open Source Driver Assistance for Your Car

**Transform your driving experience with openpilot, an open-source, community-driven driver assistance system that upgrades your car's capabilities.** [Visit the original repo on GitHub](https://github.com/commaai/openpilot)

**Key Features:**

*   **Enhanced Driver Assistance:** Adds advanced features like adaptive cruise control and lane keeping to supported vehicles.
*   **Wide Car Compatibility:** Works with over 300+ supported car models.
*   **Community-Driven Development:** Benefit from a collaborative development model with active contributions.
*   **Easy Installation:** Simple setup using a comma 3/3X device and a car harness.
*   **Regular Updates:** Stay up-to-date with new features and improvements through various branches.
*   **Data-Driven Improvement:** Uploaded driving data helps train and refine the system for better performance.
*   **Active Community:** Connect with other users and developers on Discord.

## Getting Started

To use openpilot, you'll need:

1.  **Supported Device:** A comma 3/3X ([comma.ai/shop](https://comma.ai/shop/comma-3x)).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai`.
3.  **Supported Car:** Check [the supported car list](docs/CARS.md) to confirm compatibility.
4.  **Car Harness:** You'll need a [car harness](https://comma.ai/shop/car-harness) to connect your device.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

Choose the branch that best suits your needs:

*   `release3`:  Openpilot's stable release branch. (`openpilot.comma.ai`)
*   `release3-staging`: Staging branch to get new releases slightly early. (`openpilot-test.comma.ai`)
*   `nightly`: Bleeding edge development branch, may be unstable. (`openpilot-nightly.comma.ai`)
*   `nightly-dev`:  Nightly branch with experimental features for some cars. (`installer.comma.ai/commaai/nightly-dev`)
*  `secretgoodopenpilot`:  Preview branch from the autonomy team. (`installer.comma.ai/commaai/secretgoodopenpilot`)

## Contributing & Community

Get involved with openpilot:

*   Join the [community Discord](https://discord.comma.ai)
*   Read the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Find code documentation at https://docs.comma.ai
*   Learn more on the [community wiki](https://github.com/commaai/openpilot/wiki)
*   View [comma's open positions](https://comma.ai/jobs#open-positions) & [bounties](https://comma.ai/bounties)

## Safety and Testing

*   openpilot follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.
*   Software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) are run on every commit.
*   Safety model code is in panda (C) and has [code rigor](https://github.com/commaai/panda#code-rigor).
*   panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Hardware-in-the-loop Jenkins test suite.
*   panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing on 10 comma devices.

<details>
<summary>License and Data</summary>

**MIT License:** openpilot is released under the MIT license.

**User Data:** Driving data is uploaded to improve the system; data collection can be disabled.
By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).

**Disclaimer:**
**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
</details>