# openpilot: The Open Source Driver Assistance System

**Upgrade your car's driver assistance system with openpilot, an open-source operating system for robotics that's revolutionizing the driving experience.** ([View the original repository](https://github.com/commaai/openpilot))

## Key Features:

*   **Enhanced Driver Assistance:** Provides advanced features like adaptive cruise control, lane keeping assist, and more.
*   **Wide Vehicle Compatibility:** Supports over 300+ car models, with new models continuously being added.
*   **Open Source & Community Driven:** Benefit from a collaborative development environment, with contributions from comma.ai and the open source community.
*   **Easy Installation:** Simple setup using a comma 3X device and openpilot software.
*   **Continuous Improvement:** Data-driven development, with frequent updates and improvements to performance and safety.
*   **Safety Focused:** Adheres to ISO26262 guidelines, with rigorous testing and safety measures.
*   **User Data Transparency:** Openpilot allows you to control how your data is used, with clear privacy policies.

## Getting Started:

To utilize openpilot, you'll need:

1.  **comma 3X Device:** Available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai` within the comma 3X setup.
3.  **Compatible Car:** Ensure your vehicle is on the [list of supported cars](docs/CARS.md).
4.  **Car Harness:** You'll need a [car harness](https://comma.ai/shop/car-harness) to connect the comma 3X to your car.

Detailed instructions for installation can be found at [comma.ai/setup](https://comma.ai/setup).

## Development & Contribution:

openpilot thrives on community contributions. Get involved!

*   Join the [community Discord](https://discord.comma.ai)
*   Review the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Read the code documentation at https://docs.comma.ai
*   Check out the [community wiki](https://github.com/commaai/openpilot/wiki)

## Branches:

*   **`release3`:** `openpilot.comma.ai` - The stable release branch.
*   **`release3-staging`:** `openpilot-test.comma.ai` - Staging branch for early access to new releases.
*   **`nightly`:** `openpilot-nightly.comma.ai` - Bleeding edge development branch (may be unstable).
*   **`nightly-dev`:** `installer.comma.ai/commaai/nightly-dev` - Experimental development features for some cars.

## Safety and Testing:

*   openpilot follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines. See [SAFETY.md](docs/SAFETY.md) for more information.
*   Software-in-the-loop tests run on every commit.
*   The safety model is enforced in panda (written in C).
*   Extensive testing is performed, including hardware-in-the-loop tests.

## Licensing and Data:

*   openpilot is released under the [MIT License](LICENSE).
*   By default, openpilot uploads driving data to improve the system (can be disabled). See the [Privacy Policy](https://comma.ai/privacy) for details.