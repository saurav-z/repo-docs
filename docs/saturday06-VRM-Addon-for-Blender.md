# VRM Add-on for Blender - Import, Export, and Create VRM Models

**Bring your VRM models to life within Blender with this powerful and versatile add-on!**  [View on GitHub](https://github.com/saturday06/VRM-Addon-for-Blender)

This add-on empowers Blender users to seamlessly work with VRM (Virtual Reality Model) files, enabling import, export, and advanced customization.

## Key Features:

*   **Import VRM Models:** Easily bring existing VRM models into Blender for editing and animation.
*   **Export VRM Models:** Export your Blender creations as VRM files, ready for use in compatible VR/AR applications and platforms.
*   **VRM Humanoid Support:** Adds support for humanoid rigs making VRM rigging easier.
*   **MToon Shader Configuration:** Configure and customize the popular MToon shader for anime-style visuals.
*   **PBR Material Creation:** Support for physics based rendering material creation.
*   **Animation Support:** Import and export VRM animations.
*   **Scripting API:** Automate tasks using Python scripts.

## Download

*   **Blender 4.2 or later:** [Blender Extensions Platform](https://extensions.blender.org/add-ons/vrm)
*   **Blender 2.93 to 4.1:** [Official Site](https://vrm-addon-for-blender.info)

## Tutorials

Explore these tutorials to get started:

| Installation                                                                                   | Create Simple VRM                                                                                    | Create Humanoid VRM                                                                                  |
| :---------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
| Create Physics-Based Material                                                                   | Create Anime-Style Material                                                                             | Automation with Python Scripts                                                                         |
| <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a> |
| VRM Animation                                                                                 | Development How-To                                                                                    |                                                                                                        |
| <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>     | <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>     |                                                                                                        |

## Development

The core add-on code resides in the [`src/io_scene_vrm`](https://github.com/saturday06/VRM-Addon-for-Blender/tree/main/src/io_scene_vrm) directory.

### Setting up a Development Link

Create a symbolic link to the add-on source directory in Blender's add-ons directory for easy testing and development. Instructions for different operating systems are provided below. Refer to the [development environment setup documentation](https://vrm-addon-for-blender.info/en/development?locale_redirection) for advanced development tasks.

#### Linux

```sh
blender_version=4.5
mkdir -p "$HOME/.config/blender/$blender_version/extensions/user_default"
ln -Ts "$PWD/src/io_scene_vrm" "$HOME/.config/blender/$blender_version/extensions/user_default/vrm"
```

#### macOS

```sh
blender_version=4.5
mkdir -p "$HOME/Library/Application Support/Blender/$blender_version/extensions/user_default"
ln -s "$PWD/src/io_scene_vrm" "$HOME/Library/Application Support/Blender/$blender_version/extensions/user_default/vrm"
```

#### Windows PowerShell

```powershell
$blenderVersion = 4.5
New-Item -ItemType Directory -Path "$Env:APPDATA\Blender Foundation\Blender\$blenderVersion\extensions\user_default" -Force
New-Item -ItemType Junction -Path "$Env:APPDATA\Blender Foundation\Blender\$blenderVersion\extensions\user_default\vrm" -Value "$(Get-Location)\src\io_scene_vrm"
```

### Development Link Setup for Older Blender Versions (4.1.1 or earlier)

#### Linux

```sh
blender_version=4.5
mkdir -p "$HOME/.config/blender/$blender_version/scripts/addons"
ln -Ts "$PWD/src/io_scene_vrm" "$HOME/.config/blender/$blender_version/scripts/addons/io_scene_vrm"
```

#### macOS

```sh
blender_version=4.5
mkdir -p "$HOME/Library/Application Support/Blender/$blender_version/scripts/addons"
ln -s "$PWD/src/io_scene_vrm" "$HOME/Library/Application Support/Blender/$blender_version/scripts/addons/io_scene_vrm"
```

#### Windows PowerShell

```powershell
$blenderVersion = 4.5
New-Item -ItemType Directory -Path "$Env:APPDATA\Blender Foundation\Blender\$blenderVersion\scripts\addons" -Force
New-Item -ItemType Junction -Path "$Env:APPDATA\Blender Foundation\Blender\$blenderVersion\scripts\addons\io_scene_vrm" -Value "$(Get-Location)\src\io_scene_vrm"