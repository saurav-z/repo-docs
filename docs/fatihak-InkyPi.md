# InkyPi: Your Customizable E-Ink Display for a Distraction-Free Digital Experience

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Clock Display" />

**Transform your Raspberry Pi into a beautiful, low-power display with InkyPi, showcasing the information that matters most to you.**  [View the original repo](https://github.com/fatihak/InkyPi).

## Key Features

*   **Eye-Friendly Display:** Experience crisp, paper-like visuals with no glare or distracting backlights.
*   **Web-Based Configuration:** Easily set up and customize your display from any device on your network.
*   **Minimalist & Focused:** Eliminate notifications and distractions, focusing on the content you choose.
*   **Simple Setup:** Get started quickly with easy installation and configuration suitable for all skill levels.
*   **Open Source & Customizable:** Modify, extend, and create your own plugins to tailor InkyPi to your needs.
*   **Scheduled Content Playlists:** Display different content at designated times with flexible scheduling.

## Plugins Available

InkyPi offers a range of plugins to display a variety of information:

*   Image Upload: Display your own custom images.
*   Daily Newspaper/Comic: Get your daily dose of news and comics.
*   Clock: Choose from a variety of customizable clock faces.
*   AI Image/Text: Generate dynamic content using OpenAI's models.
*   Weather: View current conditions and multi-day forecasts.
*   Calendar: Visualize your calendar from various providers.

Explore more plugins and create your own!  See [Building InkyPi Plugins](./docs/building_plugins.md) for documentation on creating custom plugins.

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W)
    *   Recommended: 40-pin Pre-Soldered Header
*   **MicroSD Card:** (min 8 GB) - [Example](https://amzn.to/3G3Tq9W)
*   **E-Ink Display:**
    *   **Pimoroni Inky Impression:**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT:** [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color: [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for more models.  Note that some models like the IT8951 based displays are not supported.
*   **Optional:** Picture Frame or 3D Stand - Explore community designs in [community.md](./docs/community.md).

**Affiliate Disclosure:** The links above are affiliate links. I may earn a commission from qualifying purchases made through them, at no extra cost to you, which helps maintain and develop this project.

## Installation Guide

Follow these steps to install InkyPi:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the installation script (using sudo):**

    *   For **Inky Impression/wHAT displays**:
        ```bash
        sudo bash install/install.sh
        ```
    *   For **Waveshare displays**: Replace `<waveshare device model>` with your display's model (e.g., `epd7in3f`):
        ```bash
        sudo bash install/install.sh -W <waveshare device model>
        ```

After installation, reboot your Raspberry Pi. Your display will show the InkyPi splash screen.

**Important Notes:**

*   `sudo` privileges are required for installation.  Use a fresh Raspberry Pi OS installation to avoid conflicts.
*   The script automatically enables the necessary SPI and I2C interfaces.
*   For detailed instructions, including how to image your microSD card, see [installation.md](./docs/installation.md) or watch [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating InkyPi

Keep your InkyPi up-to-date:

1.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch the latest changes:**
    ```bash
    git pull
    ```
3.  **Run the update script (using sudo):**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstalling InkyPi

Remove InkyPi with this command:

```bash
sudo bash install/uninstall.sh
```

## Roadmap & Future Development

InkyPi is constantly evolving!  Future features include:

*   More plugins.
*   Modular layouts.
*   Button support with customizable actions.
*   Improved Web UI on mobile.

Join the community and influence future development!  Check out the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi) to see upcoming features and vote.

## Waveshare Display Compatibility

InkyPi also supports Waveshare e-Paper displays.  **Note that IT8951-based displays are not supported, and screens smaller than 4 inches are not recommended due to resolution.**

If your Waveshare display model has a driver in their [Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd), it is likely compatible.  Use the `-W` option in the installation script to specify your display model.

## License

Distributed under the [GPL 3.0 License](./LICENSE).

This project includes fonts and icons with separate licensing and attribution requirements. See [Attribution](./docs/attribution.md) for details.

## Troubleshooting & Support

Check the [troubleshooting guide](./docs/troubleshooting.md).  If you need help, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Known Issue (Pi Zero W):** See the troubleshooting guide for known installation issues on the Pi Zero W.

## Sponsorship & Support

Support InkyPi's development:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Inspired by and with contributions from:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)