# InkyPi: Your Customizable E-Ink Dashboard 

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Display" width="600"/>

**[View the InkyPi Repository on GitHub](https://github.com/fatihak/InkyPi)**

Transform your Raspberry Pi and an E-Ink display into a stylish and energy-efficient information hub with InkyPi!

**Key Features:**

*   **Eye-Friendly Display:** Enjoy crisp, paper-like visuals with no glare or backlighting, perfect for any environment.
*   **Web-Based Control:** Easily configure and update your display from any device on your network using a user-friendly web interface.
*   **Minimalist & Focused:** Eliminate distractions with a display that offers no LEDs, noise, or notifications, just the content you need.
*   **Beginner-Friendly Setup:** Get started quickly with simple installation and configuration suitable for makers of all levels.
*   **Open-Source & Customizable:** Modify, extend, and create your own plugins to tailor InkyPi to your specific needs.
*   **Scheduled Playlists:** Display different plugins at designated times for a dynamic and personalized experience.

**Plugins:**

*   Image Upload: Display any image from your browser.
*   Daily Newspaper/Comic: Show daily comics and front pages of major newspapers.
*   Clock: Customizable clock faces.
*   AI Image/Text: Generate images and dynamic text with OpenAI.
*   Weather: Current conditions and multi-day forecasts with a custom layout.
*   Calendar: Visualize calendars from Google, Outlook, or Apple Calendar.

And more plugins are always being added!  See [Building InkyPi Plugins](./docs/building_plugins.md) for info on creating your own.

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W)
    *   Recommended: 40-pin Pre-soldered Header
*   **MicroSD Card:** (min 8 GB) - like [this one](https://amzn.to/3G3Tq9W)
*   **E-Ink Display:**
    *   **Pimoroni Inky Impression:**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT:**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models. Note that some models like the IT8951 based displays are not supported. See later section on [Waveshare e-Paper](#waveshare-display-support) compatibilty for more information.
*   **Picture Frame or Stand:**
    *   See [community.md](./docs/community.md) for examples.

**Note:** Affiliate links may be present.

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
    *   For **Inky Impression/wHAT displays:**
        ```bash
        sudo bash install/install.sh
        ```
    *   For **Waveshare displays:**
        ```bash
        sudo bash install/install.sh -W <waveshare device model>
        ```
        (Replace `<waveshare device model>` with your specific model, e.g., `epd7in3f`)

After installation, reboot your Raspberry Pi.

*   Installation requires `sudo` privileges.  A fresh install of Raspberry Pi OS is recommended to avoid conflicts.
*   The script enables SPI and I2C interfaces automatically.

For detailed instructions, see [installation.md](./docs/installation.md) or [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Update

1.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch the Latest Changes:**
    ```bash
    git pull
    ```
3.  **Run the Update Script:**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstall

```bash
sudo bash install/uninstall.sh
```

## Roadmap

*   More Plugins
*   Modular Layouts
*   Button Support
*   Improved Mobile Web UI

See the [Trello board](https://trello.com/b/SWJYWqe4/inkypi) to track progress and vote on new features!

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays.  Displays based on the **IT8951 controller are NOT supported**, and screens smaller than 4 inches are not recommended.

To use a Waveshare display, specify your display model during installation with the `-W` flag. The script will install the necessary drivers.

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE).
Separate licensing applies to fonts and icons; see [Attribution](./docs/attribution.md).

## Troubleshooting

Check the [troubleshooting guide](./docs/troubleshooting.md). For issues, create an issue on [GitHub Issues](https://github.com/fatihak/InkyPi/issues).

Note known issues during Pi Zero W installation [here](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation).

## Sponsoring

Support InkyPi's development:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)