# InkyPi: Your Customizable E-Ink Display for a Paper-Like Experience

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Displaying a Clock" width="400"/>

**Transform your Raspberry Pi into a sleek, low-power, and customizable information display with InkyPi!** [Visit the original repo](https://github.com/fatihak/InkyPi)

## Key Features

*   **Eye-Friendly Display:** Experience crisp, minimalist visuals with a paper-like aesthetic thanks to the E-Ink display, eliminating glare and backlights.
*   **Web-Based Configuration:** Effortlessly update and configure your display from any device on your network using a simple web interface.
*   **Minimalist & Distraction-Free:** Focus on the information that matters without LEDs, noise, or notifications.
*   **Easy Setup:** Beginner-friendly installation and configuration process.
*   **Open-Source & Customizable:** Modify, customize, and create your own plugins to tailor InkyPi to your needs.
*   **Scheduled Playlists:** Set up playlists to display different content at specific times.

## Plugins

InkyPi offers a range of plugins to display various types of information, with more on the way!

*   Image Upload: Display any image from your browser.
*   Daily Newspaper/Comic: View daily comics and front pages of major newspapers.
*   Clock: Customizable clock faces.
*   AI Image/Text: Generate images and dynamic text using OpenAI's models.
*   Weather: Display current weather conditions and forecasts.
*   Calendar: Visualize your calendar from Google, Outlook, or Apple Calendar.

For information on creating custom plugins, see [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W) -  Pre-soldered header is recommended.
*   **MicroSD Card:** (min 8 GB) - [Recommended Card](https://amzn.to/3G3Tq9W)
*   **E-Ink Display:**
    *   **Inky Impression by Pimoroni**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Inky wHAT by Pimoroni**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays**
        *   Spectra 6 (E6) Full Color
            *   [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126)
            *   [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126)
            *   [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White
            *   [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126)
            *   [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models.  **Note:** IT8951 based displays are **not supported**. Refer to [Waveshare e-Paper Support](#waveshare-display-support) for details.
*   **Picture Frame or 3D Stand:** Find community-made designs and builds in [community.md](./docs/community.md).

**Disclosure:** Affiliate links are provided. Purchases through these links may earn a commission at no extra cost to you, supporting the project's development.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the Installation Script:**
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```
    *   **-W \<waveshare device model\>:**  Use this *only* for Waveshare displays. Replace `<waveshare device model>` with your display's model (e.g., `epd7in3f`).

    **Examples:**

    *   Inky displays:
        ```bash
        sudo bash install/install.sh
        ```
    *   Waveshare displays:
        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```

After installation, reboot your Raspberry Pi. The display will then show the InkyPi splash screen.

**Important Notes:**

*   The installation script requires `sudo` privileges.  Start with a fresh Raspberry Pi OS installation to avoid potential conflicts.
*   The script automatically enables required SPI and I2C interfaces.

For detailed installation instructions, including how to set up Raspberry Pi OS, see [installation.md](./docs/installation.md) and this [YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Update

Keep your InkyPi up-to-date:

1.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch Latest Changes:**
    ```bash
    git pull
    ```
3.  **Run the Update Script:**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstall

To remove InkyPi:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

*   New Plugins
*   Modular Layouts
*   Button Support
*   Improved Web UI for Mobile Devices

Check the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi) for future developments and to vote on new features.

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays. Refer to [Waveshare's Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd) for model-specific drivers.

**Important Notes:**

*   **IT8951-based displays are NOT supported.**
*   Displays smaller than 4 inches are not recommended due to resolution limitations.
*   If your display model has a driver in the linked repository, it will likely be compatible.
*   Use the `-W` option with the installation script, specifying your Waveshare display model (without the `.py` extension). The script will handle the driver installation.

## License

Distributed under the GPL 3.0 License.  See [LICENSE](./LICENSE) for details.

Includes fonts and icons with separate licensing and attribution. See [Attribution](./docs/attribution.md) for details.

## Troubleshooting

See the [troubleshooting guide](./docs/troubleshooting.md).  If you still need help, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Known Issues:** There are known issues during installation on a Pi Zero W.  See the [troubleshooting guide](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) for more details.

## Sponsoring

Support InkyPi's development:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Inspired by and with help from:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)