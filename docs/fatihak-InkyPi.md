# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Display" width="500"/>

**Transform your Raspberry Pi into a stylish and energy-efficient display with InkyPi!**  [View the original repository](https://github.com/fatihak/InkyPi)

InkyPi is an open-source project that lets you display the information you need on a beautiful E-Ink screen. Designed for simplicity and flexibility, it's perfect for makers of all levels.

## Key Features

*   **Paper-Like Aesthetic:** Enjoy crisp, minimalist visuals that are easy on the eyes with no glare or backlight.
*   **Web-Based Configuration:** Easily manage and customize your display from any device on your network.
*   **Minimize Distractions:** No LEDs, noise, or notifications—just the content you care about.
*   **Easy Setup:** Simple installation and configuration, perfect for beginners.
*   **Open Source:** Modify, customize, and create your own plugins to fit your needs.
*   **Scheduled Playlists:** Display different content at designated times.

## Plugins

InkyPi offers a growing library of plugins to display a wide range of information:

*   **Image Upload:** Display any image from your browser.
*   **Daily Newspaper/Comic:** View daily comics and front pages of major newspapers.
*   **Clock:** Customize your clock faces.
*   **AI Image/Text:** Generate images and dynamic text using OpenAI's models.
*   **Weather:** Display current weather conditions and forecasts.
*   **Calendar:** Visualize your calendar from Google, Outlook, or Apple Calendar.

More plugins are coming soon!  See [Building InkyPi Plugins](./docs/building_plugins.md) for documentation on creating your own.

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W)
    *   Recommended: 40-pin Pre-Soldered Header
*   **MicroSD Card:** (min 8 GB) - [Example](https://amzn.to/3G3Tq9W)
*   **E-Ink Display:**
    *   **Pimoroni Inky Impression:**
        *   [13.3 Inch](https://collabs.shop/q2jmza)
        *   [7.3 Inch](https://collabs.shop/q2jmza)
        *   [5.7 Inch](https://collabs.shop/ns6m6m)
        *   [4 Inch](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT:**
        *   [4.2 Inch](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color: [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126), [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   More models at [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or their [Amazon store](https://amzn.to/3HPRTEZ).  **Note:** IT8951-based displays are *not* supported.  See the [Waveshare Display Support](#waveshare-display-support) section for compatibility details.
*   **Picture Frame/Stand:**  See [community.md](./docs/community.md) for examples and community submissions.

**Affiliate Disclosure:**  Some links are affiliate links, supporting the project at no extra cost to you.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the installation script:**

    For Inky displays:
    ```bash
    sudo bash install/install.sh
    ```

    For Waveshare displays, specify your model (e.g., `epd7in3f`):
    ```bash
    sudo bash install/install.sh -W epd7in3f
    ```

    The script requires `sudo` privileges.  Reboot your Raspberry Pi after installation. See [installation.md](./docs/installation.md) for detailed instructions, including how to image your microSD card.

## Updating InkyPi

1.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch the latest changes:**
    ```bash
    git pull
    ```
3.  **Run the update script:**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstalling InkyPi

```bash
sudo bash install/uninstall.sh
```

## Roadmap & Future Development

InkyPi is constantly being improved!  Here's what's planned:

*   More Plugins!
*   Modular layouts for flexible content display.
*   Support for buttons and custom actions.
*   Improved Web UI for mobile devices.

See the [Trello board](https://trello.com/b/SWJYWqe4/inkypi) to explore upcoming features and vote on your favorites!

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays.  To install for your waveshare display: use the `-W` flag and your display's model name as the argument.

*   **Supported Models:**  Check the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd) for driver compatibility.
*   **Important Notes:** IT8951 controller-based displays are *not* supported.  Screens smaller than 4 inches are *not* recommended.

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE) for more details.

## Attribution

This project uses fonts and icons with separate licensing.  See [Attribution](./docs/attribution.md) for details.

## Troubleshooting & Support

*   Check the [troubleshooting guide](./docs/troubleshooting.md).
*   Report issues on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.
*   **Pi Zero W Users:**  Refer to the troubleshooting guide for known installation issues.

## Sponsoring

Support the InkyPi project!

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Check out these related projects:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)