# openpilot: Drive Smarter, Safer with Open Source Autonomous Driving

Openpilot, the leading open-source driving platform, transforms your car into a smart vehicle with advanced driver-assistance features, supporting over 300+ car models. [See the original repository](https://github.com/commaai/openpilot).

## Key Features:

*   **Autonomous Driving:** Adds features like lane-keeping assist, adaptive cruise control, and automatic lane changes.
*   **Broad Vehicle Compatibility:** Supports over 300+ car models, with new cars and features added frequently.
*   **Open Source & Community Driven:** Benefit from continuous improvements and contributions from a vibrant community.
*   **Easy Installation:** Offers a straightforward setup using a comma 3X device.
*   **Data-Driven Improvement:** Utilizes data to improve and refine the platform.

## Getting Started

To use openpilot, you'll need:

1.  **A Supported Device:** A comma 3X ([comma.ai/shop](https://comma.ai/shop/comma-3x)).
2.  **Software:** Install openpilot via the URL: `openpilot.comma.ai` on your comma 3X.
3.  **A Supported Car:** Check for compatibility [here](docs/CARS.md).
4.  **Car Harness:** Required for connecting the comma 3X to your car ([comma.ai/shop/car-harness](https://comma.ai/shop/car-harness)).

Detailed setup instructions are available [here](https://comma.ai/setup).

## Branches

*   **`release3`:** `openpilot.comma.ai` - The official release branch.
*   **`release3-staging`:** `openpilot-test.comma.ai` -  Get new releases a bit earlier.
*   **`nightly`:** `openpilot-nightly.comma.ai` - The latest development branch (may be unstable).
*   **`nightly-dev`:** `installer.comma.ai/commaai/nightly-dev` - Nightly with experimental features.

## Contribute

Openpilot thrives on community contributions. Join us!

*   **Community:**  [Community Discord](https://discord.comma.ai)
*   **Contribute:** [Contributing Docs](docs/CONTRIBUTING.md)
*   **Tools:** Explore the [openpilot tools](tools/)
*   **Documentation:** Find code documentation at [https://docs.comma.ai](https://docs.comma.ai)
*   **Wiki:**  [Community Wiki](https://github.com/commaai/openpilot/wiki)
*   **Jobs/Bounties:** Interested in getting paid to work on openpilot? See [comma.ai/jobs#open-positions](https://comma.ai/jobs#open-positions) and [comma.ai/bounties](https://comma.ai/bounties).

## Safety and Testing

Openpilot prioritizes safety:

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines; see [SAFETY.md](docs/SAFETY.md).
*   Features software-in-the-loop tests on every commit ([.github/workflows/selfdrive_tests.yaml](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)).
*   Safety model enforced by code written in C within [panda](https://github.com/commaai/panda#code-rigor).
*   Includes software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Hardware-in-the-loop Jenkins test suite internally.
*   Continuous testing with comma devices replaying routes.

<details>
<summary>MIT License</summary>

[MIT License details as provided in the original README.]
</details>

<details>
<summary>User Data and Comma Account</summary>

[User Data and Comma Account information as provided in the original README.]
</details>