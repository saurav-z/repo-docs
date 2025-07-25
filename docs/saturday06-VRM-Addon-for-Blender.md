# VRM Add-on for Blender

[![CI status](https://github.com/saturday06/VRM-Addon-for-Blender/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/saturday06/VRM-Addon-for-Blender/actions) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Enhance your Blender workflow with the VRM Add-on, enabling seamless import, export, and manipulation of VRM models.** This add-on provides essential functionality for VRM-related tasks within Blender.

[View the original repository on GitHub](https://github.com/saturday06/VRM-Addon-for-Blender)

## Key Features

*   **VRM Import/Export:** Easily import and export VRM files, facilitating smooth model transfer.
*   **VRM Humanoid Support:**  Adds and manages VRM humanoid rigs, streamlining character setup.
*   **MToon Shader Integration:**  Allows for easy application and customization of the MToon shader, perfect for anime-style rendering.
*   **PBR Material Creation:** Create physically-based rendering materials.
*   **Animation Support:**  Integrate VRM animations into your Blender scenes.
*   **Scripting API:** Automate tasks using Python scripts.

## Download

Get the VRM Add-on for your Blender version:

*   **Blender 4.2 or later:**
    *   [🛠️ **Blender Extensions Platform**](https://extensions.blender.org/add-ons/vrm)
*   **Blender 2.93 to 4.1:**
    *   [🌐 **The Official Site**](https://vrm-addon-for-blender.info)

## Tutorials

Explore helpful guides for getting started:

| Installation | Create Simple VRM | Create Humanoid VRM |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
| Physics Based Material | Create Anime Style Material | Automation with Python Scripts |
| <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a> |     <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a>     |        <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a>        |
| VRM Animation | Development How-To | |
|    <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>    |         <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>         | |

## Development

The add-on's source code resides in the `main` branch, with the core functionality found in the `src/io_scene_vrm` folder.  For advanced development, consider using [astral.sh/uv](https://docs.astral.sh/uv/) and consult the [development tutorial](https://vrm-addon-for-blender.info/en/development?locale_redirection).
```