# MAVLink: The Lightweight Communication Protocol for Drones and Ground Control Stations

MAVLink is a powerful, open-source message library designed for seamless communication between unmanned aerial vehicles (UAVs) and ground control stations. Explore the original repository on [GitHub](https://github.com/mavlink/mavlink).

## Key Features of MAVLink

*   **Lightweight and Efficient:** MAVLink is optimized for resource-constrained environments, perfect for applications with limited bandwidth, RAM, and flash memory.
*   **Header-Only Library:**  MAVLink utilizes a header-only approach for easy integration into your projects.
*   **Cross-Platform Compatibility:**  Supports multiple programming languages, including C, Python, and more.
*   **XML-Based Message Definitions:** Defines message sets (dialects) in XML files, providing flexibility and customization.
*   **Widely Adopted:**  Field-proven and deployed in numerous products, serving as a key interoperability interface.
*   **Extensive Documentation:** Comprehensive documentation and examples available to get you started quickly.

## Getting Started with MAVLink

### Prerequisites

*   Python 3
*   `pip` (Python package installer)

### Installation (Ubuntu LTS 20.04/22.04 Example)

1.  **Install Dependencies:**
    ```bash
    sudo apt install python3-pip
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/mavlink/mavlink.git --recursive
    cd mavlink
    ```

3.  **Install Python Dependencies:**
    ```bash
    python3 -m pip install -r pymavlink/requirements.txt
    ```

### Generating C Headers

Generate C headers for `message_definitions/v1.0/common.xml`:

```bash
python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
```

### Using with CMake

1.  **Install Headers Locally:**
    ```bash
    cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
    cmake --build build --target install
    ```

2.  **Include in `CMakeLists.txt`:**
    ```cmake
    find_package(MAVLink REQUIRED)

    add_executable(my_program my_program.c)

    target_link_libraries(my_program PRIVATE MAVLink::mavlink)
    ```

3.  **Configure CMake:**
    ```bash
    cd ../my_program
    cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
    ```

For a complete example, refer to the `examples/c` directory.

**Note:** While using `target_link_libraries` in CMake, MAVLink is a header-only library, so no linking occurs.

## Further Resources

*   [Documentation/Website](https://mavlink.io/en/)
*   [Discussion/Support](https://mavlink.io/en/#support)
*   [Contributing](https://mavlink.io/en/contributing/contributing.html)
*   [License](https://mavlink.io/en/#license)
*   [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/)
*   [Installing the MAVLink Toolchain](https://mavlink.io/en/getting_started/installation.html)
*   [Generating MAVLink Libraries](https://mavlink.io/en/getting_started/generate_libraries.html)
*   [Using MAVLink Libraries](https://mavlink.io/en/getting_started/use_libraries.html)