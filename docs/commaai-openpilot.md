# openpilot: Open Source Driver Assistance for a Smarter, Safer Driving Experience

[<img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772" width="30%">](https://youtu.be/NmBfgOanCyk) [<img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c" width="30%">](https://youtu.be/VHKyqZ7t8Gw) [<img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce77dc6f63" width="30%">](https://youtu.be/SUIZYzxtMQs)

**openpilot** transforms your car into a more intelligent and safe driving machine by upgrading its driver assistance features.  This open-source software currently enhances driver assistance systems in 300+ supported vehicles.  [Visit the original repository on GitHub](https://github.com/commaai/openpilot).

**Key Features:**

*   **Enhanced Driver Assistance:** Upgrade your car's existing driver assistance systems.
*   **Open Source:** Benefit from community contributions and transparency.
*   **Wide Compatibility:** Supports over 300 car models.
*   **Continuous Improvement:** Benefit from regular updates and model training.
*   **Easy Installation:** Install easily using a comma 3/3X device.

**Getting Started:**

To begin using openpilot, you'll need:

*   **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
*   **Software:**  Install the software using the URL `openpilot.comma.ai`.
*   **Supported Car:** Verify your vehicle is on the [supported car list](docs/CARS.md).
*   **Car Harness:**  A compatible [car harness](https://comma.ai/shop/car-harness) to connect your device.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

**Software Branches:**

*   `release3`: `openpilot.comma.ai` - Release branch.
*   `release3-staging`: `openpilot-test.comma.ai` - Staging branch.
*   `nightly`: `openpilot-nightly.comma.ai` - Bleeding-edge development branch.
*   `nightly-dev`: `installer.comma.ai/commaai/nightly-dev` - Experimental development features.
*  `secretgoodopenpilot`:  `installer.comma.ai/commaai/secretgoodopenpilot` - Preview from the autonomy team.

**Contributing and Community:**

*   [Join the Community Discord](https://discord.comma.ai)
*   [Contribute to the project](docs/CONTRIBUTING.md)
*   [Explore openpilot tools](tools/)
*   [Read the documentation](https://docs.comma.ai)
*   [Visit the community wiki](https://github.com/commaai/openpilot/wiki)
*   [Check out comma's open positions](https://comma.ai/jobs#open-positions) and [bounties](https://comma.ai/bounties)

**Safety and Testing:**

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines ([SAFETY.md](docs/SAFETY.md)).
*   Software-in-the-loop tests run on every commit (.github/workflows/selfdrive_tests.yaml).
*   Safety-critical code is written in C (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   Includes software and hardware-in-the-loop safety tests.
*   Continuous testing with a large-scale testing environment.

**License & Data Usage:**

openpilot is released under the [MIT License](LICENSE).  Driving data is uploaded to comma.ai servers for model training purposes, as detailed in their [Privacy Policy](https://comma.ai/privacy). You can opt-out of data collection if desired.

***

**Disclaimer:** This is alpha-quality software for research purposes only. You are responsible for complying with all local laws and regulations. NO WARRANTY EXPRESSED OR IMPLIED.