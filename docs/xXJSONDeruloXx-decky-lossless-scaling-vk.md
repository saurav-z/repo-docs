# Enhance Steam Deck Gaming with Lossless Scaling Frame Generation

**Unlock smoother, higher framerates on your Steam Deck with this user-friendly plugin for Lossless Scaling's frame generation technology.** ([View the original repository](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk))

<p align="center">
   <img src="assets/decky-lossless-logo.png" alt="Lossless Scaling for Steam Deck Logo" width="200"/>
</p>

> **Important Note:** This is an **unofficial community plugin** for Lossless Scaling. For support, please use the [Decky Lossless Discord Channel](https://discord.gg/TwvHdVucC3).

## Key Features

*   **Simplified Installation:** Automates the setup of `lsfg-vk` (Lossless Scaling Frame Generation Vulkan layer) on your Steam Deck.
*   **Controller-Friendly UI:** Configure frame generation settings directly within SteamOS or Bazzite.
*   **Intuitive Configuration:** Easily adjust settings like FPS multiplier, flow scale, and performance mode.
*   **Real-Time Updates:** Configuration changes apply instantly, without requiring game restarts.
*   **Comprehensive Control:** Fine-tune your gaming experience with options for HDR, experimental features, and per-game settings.

## Getting Started

1.  **Purchase and Install Lossless Scaling:** Obtain the [Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/) application from Steam.
2.  **Install the Plugin:**
    *   Download the latest release ( "Lossless Scaling.zip" ) from the [releases page](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk/releases).
    *   Enable "Developer Mode" in Decky Loader settings within Game Mode.
    *   Navigate to the "Developer" tab and select "Install Plugin from Zip" to install the downloaded `.zip` file.
3.  **Use the Plugin:**
    *   Open the plugin from the Decky menu.
    *   Click "Install lsfg-vk" to set up the Vulkan layer.
    *   Configure your desired settings via the plugin's UI (FPS multiplier, flow scale, etc.).
    *   Add the launch option `~/lsfg %command%` to your game's launch options or use the plugin's "Launch Option Clipboard" button.
    *   Launch your game and enjoy the enhanced frame generation.

## Configuration Options

The plugin offers a range of settings to customize your frame generation:

### Core Settings

*   **FPS Multiplier:** 2x, 3x, or 4x frame generation.
*   **Flow Scale:** Adjusts motion estimation quality (lower for better performance, higher for better quality).
*   **Performance Mode:** Optimizes processing for improved performance.
*   **HDR Mode:** Enables HDR output for HDR-compatible games.

### Experimental Features

*   **Present Mode Override:** Force specific Vulkan presentation modes.
*   **Base FPS Limit:** Set a base framerate cap.

## Troubleshooting

*   **Frame Generation Not Working?** Ensure the correct launch option is added, the Lossless Scaling DLL is detected, and try enabling Performance Mode. Check that your game is running in fullscreen mode.
*   **Performance Issues?** Lower the Flow Scale, enable Performance Mode, or reduce the FPS multiplier. Consider using the experimental FPS limit feature.

## Plugin Breakdown

*   Automatically downloads and installs the latest lsfg-vk Vulkan layer.
*   Configures the Vulkan layer.
*   Creates a TOML configuration file with your settings.
*   Detects your Lossless Scaling DLL installation.
*   Provides an easy-to-use interface to configure frame generation settings.
*   Applies configuration changes immediately.
*   Easy uninstallation to remove all installed files.

## Credits

*   **[PancakeTAS](https://github.com/PancakeTAS/lsfg-vk)** for creating the lsfg-vk Vulkan compatibility layer.
*   **[Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/)** developers.
*   **[Deck Wizard](https://www.youtube.com/@DeckWizard)**
*   The **Decky Loader** team
*   Community contributors