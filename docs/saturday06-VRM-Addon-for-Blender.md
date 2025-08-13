# VRM Add-on for Blender: Create and Customize VRM Models in Blender

**Enhance your Blender workflow with the VRM Add-on, enabling seamless VRM model creation, import, and export.**  [Visit the GitHub Repository](https://github.com/saturday06/VRM-Addon-for-Blender)

## Key Features

*   **Import and Export VRM Files:** Easily bring VRM models into Blender for editing and export your creations in the VRM format.
*   **VRM Humanoid Support:**  Add and configure VRM Humanoid rigs for animation and character control.
*   **MToon Shader Integration:** Utilize the popular MToon shader for creating anime-style materials within Blender.
*   **PBR Material Support:** Create physically based rendering materials for realistic VRM models.
*   **Animation Tools:** Animate your VRM models directly within Blender.
*   **Python Scripting API:** Automate your VRM workflow with a powerful scripting API.
*   **Supports Blender Versions:**  Compatible with Blender 2.93 and up, and optimized for Blender 4.2 or later.

## Download

*   **For Blender 4.2 or later:**
    *   [🛠️**Blender Extensions Platform**](https://extensions.blender.org/add-ons/vrm).
*   **For Blender 2.93 to 4.1:**
    *   [🌐**The Official Site**](https://vrm-addon-for-blender.info).

## Tutorials

|                                          [Installation](https://vrm-addon-for-blender.info/en/installation?locale_redirection)                                           |                                    [Create Simple VRM](https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection)                                    |                                    [Create Humanoid VRM](https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection)                                    |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a>  | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
|                                **[Create Physics-Based Material](https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection)**                                |                                     **[Create Anime-Style Material](https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection)**                                     |                                                **[VRM Animation](https://vrm-addon-for-blender.info/en/animation?locale_redirection)**                                                 |
|  <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a>  |     <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a>     |            <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>            |
|                               **[Automation with Python Scripts](https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection)**                               |                                           **[Development How-To](https://vrm-addon-for-blender.info/en/development?locale_redirection)**                                           |                                                                                                                                                                                        |
| <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a> |        <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/development.gif"></a>        |                                                                                                                                                                                        |

## Overview

The VRM Add-on for Blender provides comprehensive functionality for VRM model creation and editing directly within Blender, including import/export, humanoid support, and MToon shader integration.  The add-on builds upon the work of the original author [@iCyP](https://github.com/iCyP), and contributions are welcome.

## Development

The add-on's code resides in the `src/io_scene_vrm` folder.  For easy development, create a symbolic link to this folder within Blender's add-ons directory to test changes. See below for instructions on creating development links for various operating systems and Blender versions.

For more advanced development, including running tests, refer to the [development environment setup documentation](https://vrm-addon-for-blender.info/en/development?locale_redirection).

### How to create a development link for Blender 4.2 or later

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

### How to create a development link for Blender 4.1.1 or earlier

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
```