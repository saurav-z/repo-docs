# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

<img src="./docs/images/inky_clock.jpg" alt="InkyPi E-Ink Display"/>

**Transform your Raspberry Pi into a sleek, low-power information hub with InkyPi, the open-source E-Ink display solution.**  [View the original repository on GitHub](https://github.com/fatihak/InkyPi)

## Key Features

*   **Eye-Friendly Display:** Enjoy crisp, minimalist visuals on a paper-like E-Ink screen with no glare or backlight.
*   **Web-Based Configuration:** Easily set up and manage your display from any device on your network using a user-friendly web interface.
*   **Distraction-Free Experience:** Focus on the content you care about with no LEDs, noise, or disruptive notifications.
*   **Simple Setup:** Get up and running quickly with easy installation and configuration, perfect for beginners and makers.
*   **Open Source & Customizable:** Modify, extend, and create your own plugins to personalize your display.
*   **Scheduled Playlists:** Display different content at various times with scheduled playlists.

## Plugins
InkyPi offers a growing collection of plugins to display various types of information:

*   Image Upload
*   Daily Newspaper/Comic
*   Clock
*   AI Image/Text Generation (using OpenAI)
*   Weather Forecasts
*   Calendar Integration

More plugins are coming soon!  For information on creating custom plugins, see the [Building InkyPi Plugins](./docs/building_plugins.md) documentation.

## Hardware Requirements

*   Raspberry Pi (4, 3, or Zero 2 W) - Pre-soldered header recommended.
*   MicroSD Card (min 8 GB) - [Example Amazon Link](https://amzn.to/3G3Tq9W)
*   E-Ink Display:
    *   Inky Impression by Pimoroni ([13.3 Inch](https://collabs.shop/q2jmza), [7.3 Inch](https://collabs.shop/q2jmza), [5.7 Inch](https://collabs.shop/ns6m6m), [4 Inch](https://collabs.shop/cpwtbh))
    *   Inky wHAT by Pimoroni ([4.2 Inch](https://collabs.shop/jrzqmf))
    *   Waveshare e-Paper Displays (See below for compatibility)
        *   Spectra 6 (E6) Full Color ([4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126), [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126))
        *   Black and White ([7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126))
    *   Find more Waveshare models on their [website](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or [Amazon store](https://amzn.to/3HPRTEZ). **IT8951-based displays are not supported.**

*   Picture Frame or 3D Stand - Browse the [community.md](./docs/community.md) file for community builds.

**Affiliate Disclosure:** *Some hardware links are affiliate links. Purchasing through these links helps support this project.*

## Installation

Follow these steps to install InkyPi:

1.  Clone the repository:
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd InkyPi
    ```
3.  Run the installation script with `sudo`:

    *   **For Inky displays:**
        ```bash
        sudo bash install/install.sh
        ```
    *   **For Waveshare displays:**  Specify your display model (e.g., `epd7in3f`):
        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```

After installation, the script will prompt you to reboot your Raspberry Pi.  Upon reboot, the InkyPi splash screen should appear.

**Important Notes:**
*   The installation script requires sudo privileges. A fresh install of Raspberry Pi OS is recommended.
*   The script automatically enables the necessary SPI and I2C interfaces.
*   Refer to [installation.md](./docs/installation.md) for detailed instructions, including how to image your microSD card.

## Update

Keep your InkyPi up-to-date:

1.  Navigate to the project directory:
    ```bash
    cd InkyPi
    ```
2.  Fetch the latest changes:
    ```bash
    git pull
    ```
3.  Run the update script with `sudo`:
    ```bash
    sudo bash install/update.sh
    ```

## Uninstall

Remove InkyPi using this command:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

InkyPi is under active development. Future plans include:

*   More plugins
*   Modular layouts
*   Button support for custom actions
*   Improved Web UI on mobile devices

Track progress and contribute ideas on the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi).

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays.  **Displays based on the IT8951 controller are not supported, and screens smaller than 4 inches are not recommended**. Consult the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd) for compatibility. When installing, use the `-W` option and your display model (e.g., `-W epd7in5b_v2`).

## License

This project is licensed under the GPL 3.0 License ([LICENSE](./LICENSE)).

This project also contains fonts and icons which have separate licensing and attribution requirements. See [Attribution](./docs/attribution.md) for details.

## Troubleshooting

Consult the [troubleshooting guide](./docs/troubleshooting.md) for solutions to common issues.  If you need further assistance, please open an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

*   **Known Pi Zero W Issues:** Check the [troubleshooting guide](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) for issues during installation on the Pi Zero W.

## Sponsoring

Support the continued development of InkyPi:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a GitHub Sponsor" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Check out these related projects:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)
```
Key improvements and SEO considerations:

*   **Clear Title and Hook:**  Uses "InkyPi" and a compelling one-sentence hook to grab attention.
*   **Keyword Optimization:** Includes relevant keywords like "E-Ink," "Raspberry Pi," "display," "customizable," and "open-source" throughout.
*   **Structured Headings:** Uses clear headings (H2) to organize information and improve readability for both humans and search engines.
*   **Bulleted Lists:** Employs bulleted lists for key features and requirements, making information easy to scan.
*   **Concise Descriptions:**  Provides brief, informative descriptions for each section.
*   **Hardware Section Improvement:**  Provides more organized hardware information and links to Waveshare displays, including the important note about IT8951.
*   **Clearer Instructions:** Installation and update steps are simplified, removing unnecessary detail for brevity.
*   **Call to Action:** Encourages users to view original repo.
*   **Internal Linking:** Includes links to relevant documentation within the project.
*   **Alt Text:**  Uses descriptive alt text for the image.
*   **Affiliate Disclosure:** Clearly states the affiliate link policy.
*   **Sponsorship/Support:** Highlights ways to support the project.
*   **Acknowledgments:** Keeps the acknowledgements.
*   **More user-friendly formatting:** Consistent use of bold and italics for better readability.