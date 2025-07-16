# Openpilot: Enhance Your Car's Driver Assistance with Open Source Technology

**Openpilot** is an open-source, community-driven project that upgrades the driver assistance systems in 300+ supported car models. For more information, visit the original repository: [https://github.com/commaai/openpilot](https://github.com/commaai/openpilot).

### Key Features:

*   **Advanced Driver-Assistance:** Enables features like adaptive cruise control, lane keeping assist, and automatic lane changes on supported vehicles.
*   **Open Source & Community Driven:** Developed by comma and a global community of contributors, ensuring continuous improvement and transparency.
*   **Wide Vehicle Compatibility:** Supports over 300 car models, with expanding support through community contributions.
*   **Easy Installation:**  Designed for use with a comma 3/3X device and a car harness.
*   **Continuous Updates:** Regular updates with new features and improvements, driven by community feedback and development.
*   **Safety Focused:**  Adheres to ISO26262 guidelines and includes rigorous testing and safety protocols.

### How to Get Started:

To use openpilot, you'll need the following:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version by using the URL `openpilot.comma.ai` during the setup process.
3.  **Supported Car:** Check the list of [supported cars](docs/CARS.md) to ensure compatibility.
4.  **Car Harness:** A compatible [car harness](https://comma.ai/shop/car-harness) to connect the comma device to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

### Branches:

*   `release3`: `openpilot.comma.ai` - Stable release branch.
*   `release3-staging`: `openpilot-test.comma.ai` - Staging branch for early access to releases.
*   `nightly`: `openpilot-nightly.comma.ai` - Bleeding-edge development branch (expect instability).
*   `nightly-dev`: `installer.comma.ai/commaai/nightly-dev` - Nightly branch with experimental features.
*   `secretgoodopenpilot`: `installer.comma.ai/commaai/secretgoodopenpilot` - Preview branch for autonomy team models.

### Contribute:

Openpilot thrives on community contributions.  Get involved:

*   Join the [community Discord](https://discord.comma.ai).
*   Review the [contributing docs](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Find code documentation at https://docs.comma.ai.
*   Learn more on the [community wiki](https://github.com/commaai/openpilot/wiki).

### Safety and Testing:

Openpilot prioritizes safety with rigorous testing and adherence to industry standards:

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Includes software-in-the-loop tests that run on every commit.
*   The safety model code is written in C (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   Features software-in-the-loop safety tests in panda.
*   Utilizes hardware-in-the-loop Jenkins test suite.
*   Runs continuous testing with multiple comma devices.

<details>
<summary>License and Disclaimer</summary>

Openpilot is released under the MIT license.

**Disclaimer:** This is alpha-quality software for research purposes only. It is not a product.  You are responsible for complying with local laws and regulations. No warranty is expressed or implied.
</details>

<details>
<summary>Data Usage</summary>

By default, openpilot uploads driving data to comma.ai servers. You can access your data through [comma connect](https://connect.comma.ai/).  You can disable data collection if you wish. Openpilot logs data including road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs. Driver-facing camera and microphone data are only logged if you explicitly opt-in. By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).
</details>