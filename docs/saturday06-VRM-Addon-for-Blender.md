# VRM Add-on for Blender: Create, Import, and Export VRM Models

**[Enhance your Blender workflow with the VRM Add-on, enabling seamless VRM model creation, import, and export directly within Blender!](https://github.com/saturday06/VRM-Addon-for-Blender)**

[![CI status](https://github.com/saturday06/VRM-Addon-for-Blender/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/saturday06/VRM-Addon-for-Blender/actions)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv)

This add-on brings comprehensive VRM (Virtual Reality Model) functionality to Blender, streamlining the creation, import, export, and manipulation of VRM models.

## Key Features

*   **VRM Import & Export:** Seamlessly import and export VRM models within Blender.
*   **Humanoid Rigging:**  Add and configure VRM Humanoid rigs for easy animation.
*   **MToon Shader Support:**  Implement MToon shaders, ideal for anime-style rendering.
*   **PBR Material Creation:**  Create realistic PBR (Physically Based Rendering) materials.
*   **Animation Support:**  Work with and create VRM animations within Blender.
*   **Scripting API:** Automate tasks with Python scripts.

## Download & Installation

*   **Blender 4.2 and later:**  Install directly from the [Blender Extensions Platform](https://extensions.blender.org/add-ons/vrm).
*   **Blender 2.93 - 4.1:** Download from the [official website](https://vrm-addon-for-blender.info).

## Tutorials

Explore our detailed tutorials to get started with VRM in Blender:

| Installation | Create Simple VRM | Create Humanoid VRM |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
| **Create Physics Based Material** | **Create Anime Style Material** | **Automation with Python Scripts** |
| <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a> |     <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a>     |        <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a>        |
| **VRM Animation** | **Development How-To** | |
|    <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>    |         <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>         |                                                                                                                                                                                        |

## Development

Contribute to the project and build your own VRM tools.

*   **Source Code:** The main source code is located in the [`main`](https://github.com/saturday06/VRM-Addon-for-Blender/tree/main) branch, specifically within the [`src/io_scene_vrm`](https://github.com/saturday06/VRM-Addon-for-Blender/tree/main/src/io_scene_vrm) folder.
*   **Efficient Development:** For easier development, create a symbolic link to the `src/io_scene_vrm` folder in your Blender addons directory.

For advanced development and testing, utilize [astral.sh/uv](https://docs.astral.sh/uv/).  Refer to the [Development Tutorial](https://vrm-addon-for-blender.info/en/development?locale_redirection) for further instructions.

```bash
git checkout main

# Blender 4.2 or later

# Linux
ln -Ts "$PWD/src/io_scene_vrm" "$HOME/.config/blender/BLENDER_VERSION/extensions/user_default/vrm"
# macOS
ln -s "$PWD/src/io_scene_vrm" "$HOME/Library/Application Support/Blender/BLENDER_VERSION/extensions/user_default/vrm"
# Windows PowerShell
New-Item -ItemType Junction -Path "$Env:APPDATA\Blender Foundation\Blender\BLENDER_VERSION\extensions\user_default\vrm" -Value "$(Get-Location)\src\io_scene_vrm"
# Windows Command Prompt
mklink /j "%APPDATA%\Blender Foundation\Blender\BLENDER_VERSION\extensions\user_default\vrm" src\io_scene_vrm

# Blender 4.1.1 or earlier

# Linux
ln -Ts "$PWD/src/io_scene_vrm" "$HOME/.config/blender/BLENDER_VERSION/scripts/addons/io_scene_vrm"
# macOS
ln -s "$PWD/src/io_scene_vrm" "$HOME/Library/Application Support/Blender/BLENDER_VERSION/scripts/addons/io_scene_vrm"
# Windows PowerShell
New-Item -ItemType Junction -Path "$Env:APPDATA\Blender Foundation\Blender\BLENDER_VERSION\scripts\addons\io_scene_vrm" -Value "$(Get-Location)\src\io_scene_vrm"
# Windows Command Prompt
mklink /j "%APPDATA%\Blender Foundation\Blender\BLENDER_VERSION\scripts\addons\io_scene_vrm" src\io_scene_vrm