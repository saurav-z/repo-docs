# openpilot: Open Source Driver Assistance for Your Car

**Upgrade your driving experience with openpilot, the open-source driving agent currently enhancing driver assistance systems in 300+ supported vehicles.**  Explore how you can enhance your car's capabilities with openpilot!  [Visit the original repository on GitHub](https://github.com/commaai/openpilot).

## Key Features

*   **Enhanced Driver Assistance:** Upgrade your car's existing driver-assistance system with features like adaptive cruise control, lane keeping, and automatic lane changes.
*   **Wide Vehicle Compatibility:** Supports over 300+ car makes and models, with continuous expansion.
*   **Open Source & Community-Driven:** Benefit from a vibrant community, open to contributions and improvements.
*   **Regular Updates:** Continuously evolving with new features, bug fixes, and vehicle support.

## Getting Started

### What You'll Need

1.  **Supported Device:** A comma 3/3X is required. Purchase one at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai` during the comma 3/3X setup.
3.  **Supported Car:** Ensure your vehicle is [supported](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) is needed to connect your comma 3/3X to your car.

### Installation

Follow the detailed instructions for [how to install the harness and device in a car](https://comma.ai/setup).

## Branches

Choose the branch that best fits your needs:

*   `release3`: `openpilot.comma.ai` (Release branch)
*   `release3-staging`: `openpilot-test.comma.ai` (Staging branch for early access)
*   `nightly`: `openpilot-nightly.comma.ai` (Bleeding edge development branch, may be unstable)
*   `nightly-dev`: `installer.comma.ai/commaai/nightly-dev` (Nightly with experimental features for some cars)
*   `secretgoodopenpilot`: `installer.comma.ai/commaai/secretgoodopenpilot` (Preview branch from the autonomy team)

## Contribute

openpilot thrives on community contributions.  Get involved by:

*   Joining the [community Discord](https://discord.comma.ai).
*   Reviewing the [contributing docs](docs/CONTRIBUTING.md).
*   Exploring the [openpilot tools](tools/).
*   Accessing code documentation at https://docs.comma.ai.
*   Consulting the [community wiki](https://github.com/commaai/openpilot/wiki) for helpful information.

## Safety and Testing

openpilot prioritizes safety through:

*   Adherence to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.  See [SAFETY.md](docs/SAFETY.md) for details.
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   C-based code for safety-critical components (panda).
*   Extensive testing, including hardware-in-the-loop and continuous replay testing.

## Licensing and Data

*   openpilot is licensed under the MIT license.
*   Data is collected by default and used to improve the system.  Users can disable data collection.  See [our Privacy Policy](https://comma.ai/privacy).

---

**Disclaimer:** *This is alpha-quality software for research purposes only. It is not a product. You are responsible for complying with all local laws and regulations.  No warranty is expressed or implied.*