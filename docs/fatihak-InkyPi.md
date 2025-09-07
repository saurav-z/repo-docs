# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

[![InkyPi Display](docs/images/inky_clock.jpg)](https://github.com/fatihak/InkyPi)

**Transform your Raspberry Pi into a stylish and low-power display with InkyPi, offering a crisp, distraction-free experience for showcasing the information you need.**

## Key Features

*   **Eye-Friendly Display:** Enjoy crisp, paper-like visuals with no glare or backlighting.
*   **Web-Based Control:** Easily configure and update your display from any device on your network via a user-friendly web interface.
*   **Minimalist Design:** Eliminate distractions with no LEDs, noise, or notifications, just the content you care about.
*   **Easy Setup:** Simple installation and configuration, perfect for beginners and makers alike.
*   **Open-Source & Customizable:** Modify, customize, and create your own plugins to suit your needs.
*   **Scheduled Playlists:** Display different plugins at designated times for dynamic information updates.

## Plugins

InkyPi supports a variety of plugins, with more on the way:

*   **Image Upload:** Display any image from your browser.
*   **Daily Newspaper/Comic:** View daily comics and front pages of major newspapers.
*   **Clock:** Customize your clock faces for displaying time.
*   **AI Image/Text:** Generate images and dynamic text using OpenAI's models.
*   **Weather:** Display current weather conditions and forecasts.
*   **Calendar:** Visualize your calendar from Google, Outlook, or Apple Calendar.

For information on creating custom plugins, see the [Building InkyPi Plugins](./docs/building_plugins.md) documentation.

## Hardware

You'll need the following hardware to get started:

*   **Raspberry Pi:** (4 | 3 | Zero 2 W) - Recommended to get a 40 pin Pre Soldered Header.
*   **MicroSD Card:** Minimum 8GB, such as [this one](https://amzn.to/3G3Tq9W).
*   **E-Ink Display:**

    *   **Inky Impression by Pimoroni:**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Inky wHAT by Pimoroni:**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color: [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) | [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) | [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) | [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models.  **Note:** IT8951 based displays are **not** supported.

*   **Picture Frame or 3D Stand:** See [community.md](./docs/community.md) for community-made designs.

**Affiliate Disclosure:**  The links provided are affiliate links, and I may earn a commission from qualifying purchases at no extra cost to you. Your support helps maintain and improve this project.

## Installation

Follow these steps to install InkyPi:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the Installation Script (with `sudo`):**

    *   **For Inky Displays:**
        ```bash
        sudo bash install/install.sh
        ```
    *   **For Waveshare Displays:** (Specify your display model)
        ```bash
        sudo bash install/install.sh -W <waveshare device model>
        ```
        (Replace `<waveshare device model>` with your Waveshare display model, e.g., `epd7in3f`)

After installation, the script will prompt you to reboot your Raspberry Pi. Upon reboot, you'll see the InkyPi splash screen.

**Important Notes:**

*   The installation script requires `sudo` privileges.
*   Start with a fresh Raspberry Pi OS installation to avoid conflicts.
*   SPI and I2C interfaces are automatically enabled.

For detailed instructions, including how to image your microSD card, see [installation.md](./docs/installation.md) and [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Update

To update your InkyPi:

1.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch Latest Changes:**
    ```bash
    git pull
    ```
3.  **Run the Update Script (with `sudo`):**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstall

To remove InkyPi:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

The InkyPi project is continuously evolving, with future plans including:

*   More plugins
*   Modular layouts
*   Button support with customizable actions
*   Improved Web UI for mobile devices

Stay updated and contribute to the project via the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi).

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays. Note that IT8951-based displays are **not** supported, and screens smaller than 4 inches are **not recommended**.

The installation script utilizes drivers from the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd).  Ensure your display model has a corresponding driver in this library. When installing, use the `-W` option and specify your model.

## License

Distributed under the GPL 3.0 License.  See [LICENSE](./LICENSE) for details.

This project uses fonts and icons with separate licensing requirements.  See [Attribution](./docs/attribution.md) for more information.

## Troubleshooting

Refer to the [troubleshooting guide](./docs/troubleshooting.md) for solutions to common issues.  For additional help, open an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Known Issues:**  There are known issues during the Pi Zero W installation; see [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation).

## Sponsoring

Support the ongoing development of InkyPi:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Sponsor" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Explore these related projects:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)

[**Visit the InkyPi GitHub Repository**](https://github.com/fatihak/InkyPi)