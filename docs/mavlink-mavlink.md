# MAVLink: The Lightweight Communication Protocol for Drones and Ground Control Stations

MAVLink is the premier open-source message library enabling seamless communication between drones, ground control stations, and other unmanned systems. You can find the original project on GitHub: [mavlink/mavlink](https://github.com/mavlink/mavlink).

## Key Features

*   **Lightweight and Efficient:** Designed for resource-constrained systems with limited bandwidth, RAM, and flash memory.
*   **Cross-Platform Compatibility:** Supports a wide range of programming languages including C, Python, and others, facilitating interoperability.
*   **Header-Only Library:** Simplifies integration and reduces dependencies.
*   **XML-Based Message Definitions:**  Uses XML files for defining message sets ("dialects"), ensuring flexibility and maintainability.
*   **Field-Proven and Widely Adopted:**  Trusted by numerous manufacturers and deployed in various drone and unmanned systems.
*   **Comprehensive Tooling:** Offers Python scripts for code generation, examples, and utilities for efficient MAVLink data handling.

## Quick Start Guide

### Generate C Headers

1.  **Install Dependencies (Ubuntu):**

    ```bash
    sudo apt install python3-pip
    ```

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/mavlink/mavlink.git --recursive
    cd mavlink
    ```

3.  **Install Python Requirements:**

    ```bash
    python3 -m pip install -r pymavlink/requirements.txt
    ```

4.  **Generate C Library:**

    ```bash
    python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
    ```

### Use from CMake

1.  **Install Headers Locally:**

    ```bash
    cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
    cmake --build build --target install
    ```

2.  **Integrate in `CMakeLists.txt`:**

    ```cmake
    find_package(MAVLink REQUIRED)
    add_executable(my_program my_program.c)
    target_link_libraries(my_program PRIVATE MAVLink::mavlink)
    ```

3.  **Configure CMake with Install Directory:**

    ```bash
    cd ../my_program
    cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
    ```

    Refer to the [examples/c](examples/c) for a full example.

*Note: MAVLink is a header-only library.*

## Resources

*   [Documentation/Website](https://mavlink.io/en/)
*   [Discussion/Support](https://mavlink.io/en/#support)
*   [Contributing](https://mavlink.io/en/contributing/contributing.html)
*   [License](https://mavlink.io/en/#license)
*   [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/)
*   [Installing the MAVLink Toolchain](https://mavlink.io/en/getting_started/installation.html)
*   [Generating MAVLink Libraries](https://mavlink.io/en/getting_started/generate_libraries.html)
*   [Using MAVLink Libraries](https://mavlink.io/en/getting_started/use_libraries.html)