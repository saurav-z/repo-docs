# VRM Add-on for Blender

[![CI status](https://github.com/saturday06/VRM-Addon-for-Blender/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/saturday06/VRM-Addon-for-Blender/actions) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Enhance your Blender workflow with the VRM Add-on, enabling seamless import, export, and creation of VRM models!**  This add-on extends Blender's functionality to support the VRM (Virtual Reality Model) format.  It allows you to easily create, import, and export VRM files, making it a must-have tool for VR/AR content creation, virtual avatars, and more.  Explore the full potential of VRM within Blender!

➡️ [Visit the GitHub Repository for the latest updates and source code](https://github.com/saturday06/VRM-Addon-for-Blender)

## Key Features

*   **VRM Import/Export:** Seamlessly import and export VRM models to and from Blender.
*   **VRM Humanoid Support:**  Add and configure VRM Humanoid rigs for animation and rigging.
*   **MToon Shader Configuration:** Utilize MToon shaders for anime-style rendering within your VRM models.
*   **Physics-Based Material:** Create physics-based material with PBR configuration.
*   **Comprehensive Tutorials:** Access detailed tutorials covering installation, VRM creation (simple & humanoid), and advanced features like material creation and scripting.
*   **Scripting API:** Automate tasks and extend the add-on's functionality with Python scripting.
*   **Active Development:** The project is actively maintained and welcomes contributions.

## Download

*   **Blender 4.2 or later:**  [Blender Extensions Platform](https://extensions.blender.org/add-ons/vrm)
*   **Blender 2.93 to 4.1:** [Official Site](https://vrm-addon-for-blender.info)

## Tutorials

[Installation](https://vrm-addon-for-blender.info/en/installation?locale_redirection) | [Create Simple VRM](https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection) | [Create Humanoid VRM](https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection)
:---: | :---: | :---:
<img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"> | <img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"> | <img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif">
[Create Physics-Based Material](https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection) | [Create Anime-Style Material](https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection) | [Automation with Python Scripts](https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection)
:---: | :---: | :---:
<img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"> | <img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"> | <img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif">
[VRM Animation](https://vrm-addon-for-blender.info/en/animation?locale_redirection) | [Development How-To](https://vrm-addon-for-blender.info/en/development?locale_redirection) |
:---: | :---: |
<img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"> | <img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"> |

## Development

The `src/io_scene_vrm` folder contains the main add-on code.  Follow the instructions below to create a development link for easy testing and contribute to the project.

### Create a Development Link

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

*   **For Blender 4.1.1 or earlier, use the following instructions instead, replacing the `extensions` path with `scripts/addons`:**

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