# openpilot: Open Source Driver Assistance System

**Upgrade your car's driving capabilities with openpilot, an open-source, community-driven driver-assistance system.** ([See the original repo](https://github.com/commaai/openpilot))

openpilot is an operating system for robotics, currently enhancing driver assistance features in over 300 supported car models.

## Key Features

*   **Enhanced Driver Assistance:** Adds advanced features like adaptive cruise control, lane keeping assist, and automatic lane changes.
*   **Open Source & Community Driven:** Benefit from continuous improvements and contributions from a vibrant community of developers.
*   **Wide Car Compatibility:** Supports a growing list of over 300 car models.
*   **Easy Installation:** Simple setup using a comma 3/3X device.
*   **Continuous Testing & Safety:** Rigorous testing, including software-in-the-loop and hardware-in-the-loop tests, to ensure safety and reliability.

## Getting Started

### What You Need

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version by entering the URL `openpilot.comma.ai` during the setup of your comma 3/3X.
3.  **Supported Car:** Verify your car is on [the supported car list](docs/CARS.md).
4.  **Car Harness:** You will also need a [car harness](https://comma.ai/shop/car-harness) to connect your comma 3/3X to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

### Available Branches

*   `release3`: Stable release branch (openpilot.comma.ai)
*   `release3-staging`: Staging branch for early releases (openpilot-test.comma.ai)
*   `nightly`: Bleeding-edge development branch (openpilot-nightly.comma.ai)
*   `nightly-dev`: Experimental development features (installer.comma.ai/commaai/nightly-dev)
*   `secretgoodopenpilot`: Preview branch (installer.comma.ai/commaai/secretgoodopenpilot)

## Contributing

openpilot thrives on community contributions. Help improve openpilot by:

*   Joining the [community Discord](https://discord.comma.ai).
*   Reviewing the [contributing docs](docs/CONTRIBUTING.md).
*   Exploring the [openpilot tools](tools/).
*   Checking out the [code documentation](https://docs.comma.ai).
*   Consulting the [community wiki](https://github.com/commaai/openpilot/wiki) for additional information.

Interested in working on openpilot professionally? Explore [comma.ai/jobs#open-positions](https://comma.ai/jobs#open-positions) for employment opportunities and [comma.ai/bounties](https://comma.ai/bounties) for bounties.

## Safety and Testing

openpilot is committed to safety:

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.
*   Features software-in-the-loop tests that run on every commit.
*   The code enforcing the safety model lives in panda and is written in C.
*   Includes extensive hardware-in-the-loop testing.

## License and Data Privacy

*   **License:** MIT ([LICENSE](LICENSE))
*   **Data:** By default, openpilot uploads driving data to improve the system. Users can disable data collection. By using openpilot, you agree to the [Privacy Policy](https://comma.ai/privacy).

***

**Disclaimer:** This is alpha-quality software for research purposes only. You are responsible for complying with local laws and regulations. NO WARRANTY EXPRESSED OR IMPLIED.