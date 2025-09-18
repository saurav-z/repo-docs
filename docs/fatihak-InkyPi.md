# InkyPi: Your Customizable E-Ink Display Powered by Raspberry Pi

[![InkyPi Display](docs/images/inky_clock.jpg)](https://github.com/fatihak/InkyPi)

**Transform your Raspberry Pi into a low-power, distraction-free display with InkyPi, showcasing the information you need with the beauty of e-ink.**

[View the InkyPi Repository on GitHub](https://github.com/fatihak/InkyPi)

## Key Features

*   **Eye-Friendly Display:** Enjoy a paper-like aesthetic with crisp, minimalist visuals, eliminating glare and backlighting.
*   **Web-Based Control:** Effortlessly configure and update your display from any device on your network using a simple web interface.
*   **Minimal Distractions:** Focus on the content that matters, with no LEDs, noise, or unwanted notifications.
*   **Easy Setup:** Get started quickly with straightforward installation and configuration, perfect for beginners and makers.
*   **Open Source & Customizable:** Modify, extend, and create your own plugins thanks to the open-source nature of the project.
*   **Scheduled Content:** Set up playlists to display different plugins at specified times.

## Available Plugins

InkyPi offers a range of plugins to display a variety of information.

*   **Image Upload:** Display any image from your browser.
*   **Daily Newspaper/Comic:** Show daily comics and front pages of major newspapers.
*   **Clock:** Customize your clock faces to display the time.
*   **AI Image/Text:** Generate images and dynamic text with prompts using OpenAI's models.
*   **Weather:** View current weather conditions and forecasts with a customizable layout.
*   **Calendar:** Visualize your calendar from Google, Outlook, or Apple Calendar.

And new plugins are always on the way!  Explore plugin development with the [Building InkyPi Plugins](docs/building_plugins.md) documentation.

## Hardware Requirements

You'll need the following components:

*   **Raspberry Pi:** (4 | 3 | Zero 2 W)
    *   A pre-soldered 40-pin header is recommended.
*   **MicroSD Card:** Minimum 8 GB (e.g., [Recommended MicroSD](https://amzn.to/3G3Tq9W))
*   **E-Ink Display:** Compatible displays include:
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
        *   Browse more models at [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or their [Amazon store](https://amzn.to/3HPRTEZ). **Note:** IT8951-based displays are not supported.

*   **Picture Frame/Stand:** Explore the [community.md](./docs/community.md) document for user-submitted 3D models and custom builds.

**Affiliate Disclosure:** Some links are affiliate links, and I may receive a commission on qualifying purchases at no extra cost to you, supporting the project's ongoing development.

## Installation Guide

Get InkyPi up and running with these steps:

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
    *   For Inky displays:
        ```bash
        sudo bash install/install.sh
        ```
    *   For Waveshare displays, use the `-W` option with your display model (e.g., `epd7in3f`):
        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```
    *   This script may take some time, and it will prompt you to reboot your Raspberry Pi.
    *   Review [installation.md](./docs/installation.md) for more details, including instructions on how to flash your microSD card and the [YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating InkyPi

Keep your InkyPi up-to-date with the following steps:

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

## Uninstalling InkyPi

Remove InkyPi from your system with this command:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

InkyPi is an active project, and here's what's planned:

*   Expanded plugin library.
*   Modular layouts for flexible customization.
*   Button support with custom action bindings.
*   Improved web UI optimized for mobile devices.

Visit the [Trello board](https://trello.com/b/SWJYWqe4/inkypi) to track progress and vote on upcoming features!

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays, but they require specific drivers.  The project has been tested on various Waveshare models. Displays with the IT8951 controller are not supported, and smaller than 4-inch screens are not recommended due to resolution limitations.

When installing, use the `-W` option with the correct model name.  The script will install the corresponding driver.

## Licensing

InkyPi is released under the GPL 3.0 License.  Refer to [LICENSE](./LICENSE) for more information.

Additional licensing for fonts and icons can be found in the [Attribution](./docs/attribution.md) document.

## Troubleshooting & Support

*   Review the [troubleshooting guide](./docs/troubleshooting.md).
*   If you need help, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.
*   **Pi Zero W Note:** Known issues can happen during installation. See [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) for additional information.

## Supporting the Project

If you find InkyPi useful, consider supporting its ongoing development:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a GitHub Sponsor" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Check out these related projects:

*   [PaperPi](https://github.com/txoof/PaperPi) - Supports Waveshare devices.
*   [InkyCal](https://github.com/aceinnolab/Inkycal) - Modular plugins.
*   [PiInk](https://github.com/tlstommy/PiInk) - Inspiration for the Flask web UI.
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display) - Alternative e-ink weather dashboard.