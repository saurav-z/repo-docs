# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

[<img src="./docs/images/inky_clock.jpg" alt="InkyPi Clock" width="500">](https://github.com/fatihak/InkyPi)

**Transform your Raspberry Pi into a low-power, distraction-free information hub with InkyPi, a customizable E-Ink display project.**

## Key Features

*   **Eye-Friendly Display:** Enjoy crisp, paper-like visuals with no glare or backlight.
*   **Web-Based Control:** Easily configure and update your display from any device on your network.
*   **Minimalist Design:** Eliminate distractions with a display free of LEDs, noise, and notifications.
*   **Simple Setup:** Perfect for beginners and makers, with straightforward installation and configuration.
*   **Open Source Flexibility:** Modify, customize, and create your own plugins to suit your needs.
*   **Scheduled Playlists:** Set up playlists to display different plugins at designated times.

## Available Plugins

InkyPi offers a variety of plugins to display the information you need:

*   **Image Upload:** Display any image you upload via your browser.
*   **Daily Newspaper/Comic:** Show daily comics and front pages of major newspapers.
*   **Clock:** Customizable clock faces to display the time.
*   **AI Image/Text:** Generate images and text from prompts using OpenAI's models.
*   **Weather:** Display current conditions and forecasts with a customizable layout.
*   **Calendar:** Visualize your calendar from Google, Outlook, or Apple Calendar.

More plugins are on the way! Learn how to build your own custom plugins by exploring the [Building InkyPi Plugins documentation](./docs/building_plugins.md).

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W) - Recommended with a 40-pin pre-soldered header.
*   **MicroSD Card:** Minimum 8GB (e.g., [Amazon Link](https://amzn.to/3G3Tq9W) - affiliate link).
*   **E-Ink Display:**
    *   **Pimoroni Inky Impression:**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT:**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color: [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) | [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) | [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) | [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models. Note that some models like the IT8951 based displays are not supported. See later section on [Waveshare e-Paper](#waveshare-display-support) compatibility for more information.
*   **Picture Frame or 3D Stand:** Explore community submissions for inspiration in [community.md](./docs/community.md).

**Affiliate Disclosure:** Some links above are affiliate links, and I may earn a commission from qualifying purchases at no extra cost to you, which helps support the project.

## Installation

Get your InkyPi display up and running with these simple steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the installation script (with sudo):**
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```

    *   **For Inky Displays:**
        ```bash
        sudo bash install/install.sh
        ```
    *   **For Waveshare Displays:**
        ```bash
        sudo bash install/install.sh -W <waveshare_display_model>
        ```
        *   Replace `<waveshare_display_model>` with your Waveshare display model (e.g., `epd7in3f`).

After installation, reboot your Raspberry Pi.  For detailed instructions, see [installation.md](./docs/installation.md).

## Updating InkyPi

Keep your InkyPi updated with the latest features and fixes:

1.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch the latest changes:**
    ```bash
    git pull
    ```
3.  **Run the update script (with sudo):**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstalling InkyPi

To remove InkyPi, use the following command:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

The InkyPi project is constantly evolving; upcoming features include:

*   More plugins.
*   Modular layouts to mix and match plugins.
*   Button support with customizable actions.
*   Improved mobile web UI.

Check out the [public Trello board](https://trello.com/b/SWJYWqe4/inkypi) to see what's next and vote on your favorite features!

## Waveshare Display Support

InkyPi also supports Waveshare e-Paper displays. Be aware that displays using the IT8951 controller are not supported, and screens smaller than 4 inches are not recommended. To install with Waveshare displays, follow the steps in the [Installation](#installation) section.
Refer to the [Waveshare e-Paper library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd) to check if your model is supported.

## License

InkyPi is distributed under the [GPL 3.0 License](./LICENSE). See [Attribution](./docs/attribution.md) for details on the licensing of included fonts and icons.

## Troubleshooting & Support

Find solutions to common issues in the [troubleshooting guide](./docs/troubleshooting.md). If you still need help, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

## Sponsoring

Help support the continued development of InkyPi!

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Explore these related projects:

*   [PaperPi](https://github.com/txoof/PaperPi) - Supporting Waveshare devices.
*   [InkyCal](https://github.com/aceinnolab/Inkycal) - Modular plugins for custom dashboards.
*   [PiInk](https://github.com/tlstommy/PiInk) - Inspired InkyPi's Flask web UI.
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display) - Alternative e-ink weather dashboard.

---

**[Back to the GitHub Repository](https://github.com/fatihak/InkyPi)**