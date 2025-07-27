# Enhance Your Steam Deck Gaming with Lossless Scaling Frame Generation

This community-made Decky plugin streamlines the installation and configuration of **lsfg-vk** on your Steam Deck, allowing you to experience smoother, more fluid gameplay with Lossless Scaling frame generation. For the original repository, check out: [github.com/xXJSONDeruloXx/decky-lossless-scaling-vk](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk).

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/B0B71HZTAX)

<p align="center">
   <img src="assets/decky-lossless-logo.png" alt="Lossless Scaling for Steam Deck Logo" width="200"/>
</p>

> **Note:** This is an **unofficial community plugin** and is not officially supported by the creators of Lossless Scaling or lsfg-vk. For support, please use the [Decky Lossless Discord Channel](https://discord.gg/TwvHdVucC3).

## Key Features

*   **Simplified Installation**: Automatically installs and configures the `lsfg-vk` Vulkan layer.
*   **User-Friendly Interface**: Easily adjust frame generation settings directly from the Decky menu.
*   **Real-Time Configuration**: Changes to settings apply immediately without game restarts.
*   **FPS Multiplier Control**: Choose between 2x, 3x, and 4x frame generation.
*   **Performance Optimization**: Fine-tune performance with Flow Scale and Performance Mode options.
*   **HDR Support**: Enable HDR mode for compatible games.
*   **Experimental Features**: Access advanced settings, including Present Mode Override and Base FPS Limit.
*   **Easy Uninstallation**: Completely removes all installed files when no longer needed.

## Installation

1.  **Download the plugin** from the [latest release](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk/releases) - Download the "Lossless Scaling.zip" file to your Steam Deck
2.  **Install manually through Decky**:
    *   In Game Mode, go to the settings cog in the top right of the Decky Loader tab
    *   Enable "Developer Mode"
    *   Go to "Developer" tab and select "Install Plugin from Zip"
    *   Select the downloaded "Lossless Scaling.zip" file

## How to Use

1.  **Purchase and install** [Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/) from Steam
2.  **Open the plugin** from the Decky menu
3.  **Click "Install lsfg-vk"** to automatically set up the lsfg-vk vulkan layer
4.  **Configure settings** using the plugin's UI - adjust FPS multiplier, flow scale, performance mode, HDR settings, and experimental features
5.  **Apply launch option** to games you want to use frame generation with:
    *   Add `~/lsfg %command%` to your game's launch options in Steam Properties
    *   Or use the "Launch Option Clipboard" button in the plugin to copy the command
6.  **Launch your game** - frame generation will activate automatically using your plugin configuration

**Note**: Configuration changes are applied in real-time and will take effect immediately without restarting your game.

## Configuration Options

### Core Settings

*   **FPS Multiplier**: 2x, 3x, or 4x frame generation
*   **Flow Scale**: Adjust motion estimation quality
*   **Performance Mode**: Lighter processing
*   **HDR Mode**: Enable for HDR output

### Experimental Features

*   **Present Mode Override**: Force specific Vulkan presentation modes.
*   **Base FPS Limit**: Set a base framerate cap.

## Troubleshooting

**Frame generation not working?**

*   Ensure you've added `LSFG_PROCESS=decky-lsfg-vk %command%` to your game's launch options
*   Check that the Lossless Scaling DLL was detected correctly in the plugin
*   Try enabling Performance Mode if you're experiencing crashes
*   Make sure your game is running in fullscreen mode for best results

**Performance issues?**

*   Lower the Flow Scale setting
*   Enable Performance Mode
*   Try reducing the FPS multiplier
*   Consider using the experimental FPS limit feature

## Credits

*   **[PancakeTAS](https://github.com/PancakeTAS/lsfg-vk)** for creating the lsfg-vk Vulkan compatibility layer
*   **[Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/)** developers for the original frame generation technology
*   **[Deck Wizard](https://www.youtube.com/@DeckWizard)** for the helpful video tutorial
*   The **Decky Loader** team for the plugin framework
*   Community contributors and testers for feedback and bug reports