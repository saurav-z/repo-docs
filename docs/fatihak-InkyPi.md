# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

<img src="./docs/images/inky_clock.jpg" alt="InkyPi E-Ink Display" width="500"/>

**Transform your Raspberry Pi into a stylish and energy-efficient display with InkyPi, the open-source solution for showcasing the information you care about.** ([Back to Original Repo](https://github.com/fatihak/InkyPi))

## Key Features

*   **Paper-Like Visuals:** Enjoy crisp, minimalist visuals with no glare or backlighting, providing a natural reading experience.
*   **Web-Based Control:** Easily update and configure your display from any device on your network via a user-friendly web interface.
*   **Distraction-Free Design:** Focus on the content you need with no LEDs, noise, or notifications.
*   **Simple Setup:** Get up and running quickly with easy installation and configuration, perfect for beginners and makers.
*   **Open-Source Flexibility:** Customize, modify, and extend InkyPi with your own plugins and creations.
*   **Scheduled Playlists:** Set up playlists to display different plugins at specified times.

## Core Features:

*   **Image Upload:** Display any image from your browser.
*   **Daily Newspaper/Comic:** Showcase daily comics and front pages from global newspapers.
*   **Clock:** Personalize your display with custom clock faces.
*   **AI Image/Text Generation:** Generate images and dynamic text via OpenAI models.
*   **Weather:** Display current and forecast weather conditions with a customizable layout.
*   **Calendar:** Visualize your Google, Outlook, or Apple Calendar with customized layouts.

...And more plugins are constantly being added!  For details on creating custom plugins, see [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W) -  A 40-pin Pre-Soldered Header is recommended.
*   **MicroSD Card:** (min 8 GB) - [Example](https://amzn.to/3G3Tq9W)
*   **E-Ink Display:**  Choose from a variety of supported displays:

    *   **Pimoroni Inky Impression:**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT:** [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) & [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) & [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) & [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   Browse the full range and find additional models at [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or the [Waveshare Amazon store](https://amzn.to/3HPRTEZ).  **Note:** IT8951-based displays are not supported.

*   **Picture Frame or 3D Stand:** Find inspiration and community-contributed designs in [community.md](./docs/community.md).

**Affiliate Disclosure:**  *As an Amazon Associate, I earn from qualifying purchases. These links help support the project at no extra cost to you.*

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the installation script:** Use `sudo` privileges.

    *   **For Inky displays:**
        ```bash
        sudo bash install/install.sh
        ```

    *   **For Waveshare displays:**
        ```bash
        sudo bash install/install.sh -W <waveshare device model>
        ```
        (Replace `<waveshare device model>` with your specific model, e.g., `epd7in3f`).

After installation, reboot your Raspberry Pi.  The display will then show the InkyPi splash screen.

**Important:**

*   Requires `sudo` privileges. Consider a fresh Raspberry Pi OS installation to avoid conflicts.
*   The script automatically enables necessary SPI and I2C interfaces.
*   See [installation.md](./docs/installation.md) for detailed instructions, including Raspberry Pi OS imaging.  There's also a helpful [YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating InkyPi

1.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch latest changes:**
    ```bash
    git pull
    ```
3.  **Run the update script:**  Use `sudo` privileges.
    ```bash
    sudo bash install/update.sh
    ```
This process ensures that updates, including code changes and dependencies, are applied without a full reinstallation.

## Uninstalling InkyPi

```bash
sudo bash install/uninstall.sh
```

## Roadmap & Future Development

The InkyPi project is actively evolving!  Future developments include:

*   More plugins and improvements to existing ones.
*   Modular layouts for mixing and matching plugins.
*   Support for buttons with customizable action bindings.
*   Improved Web UI for mobile devices.

Join the community and influence development! See the public [trello board](https://trello.com/b/SWJYWqe4/inkypi) to vote on what you'd like to see next.

## Waveshare Display Support

InkyPi also supports Waveshare e-Paper displays. These displays have slightly different configuration needs.

*   **Unsupported:** IT8951-based displays. Displays smaller than 4 inches are not recommended.
*   If a driver exists for your Waveshare model in their [Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd), it is likely compatible.
*   Specify your display model using the `-W` option during installation (e.g., `sudo bash install/install.sh -W epd7in3f`). The script will automatically download and install the required driver.

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE) for details.

This project includes fonts and icons with separate licensing requirements; see [Attribution](./docs/attribution.md) for details.

## Troubleshooting

Refer to the [troubleshooting guide](./docs/troubleshooting.md) for solutions. For issues or suggestions, please create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Known Issues (Pi Zero W):** See the [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) section in the troubleshooting guide.

## Sponsoring

Support the development of InkyPi!

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Inspired by and with thanks to:

*   [PaperPi](https://github.com/txoof/PaperPi)
    *   @txoof assisted with InkyPi's installation process
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)