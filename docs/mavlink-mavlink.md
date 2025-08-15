# MAVLink: The Lightweight Communication Protocol for Drones

**MAVLink empowers seamless communication between drones, ground control stations, and other aerial components, enabling reliable and efficient data exchange.**

[Link to Original Repo: mavlink/mavlink](https://github.com/mavlink/mavlink)

MAVLink (Micro Air Vehicle Message Marshalling Library) is a header-only message library designed for robust and bandwidth-efficient communication within the drone ecosystem. It's built upon XML-defined message sets (dialects) and Python tools, offering broad language support and optimized performance.  It's field-proven and deployed in many products where it serves as interoperability interface between components of different manufacturers.

## Key Features of MAVLink

*   **Lightweight and Efficient:** Designed for resource-constrained environments, minimizing RAM and flash memory usage.
*   **Header-Only Library:** Simplifies integration and reduces dependencies.
*   **Cross-Platform Compatibility:** Supports multiple programming languages, including C, Python, and others.
*   **XML-Based Message Definitions:** Simplifies message creation and updates.
*   **Wide Adoption:**  Used in numerous drone and ground control station applications.
*   **Highly Optimized:**  Maximizes performance for bandwidth-limited communication.
*   **Interoperability:** Ensures seamless communication between components from different manufacturers.

## Getting Started (C Example)

Here's a basic guide to generate C headers and integrate them into your project:

1.  **Install Dependencies (Ubuntu 20.04/22.04):**
    ```bash
    sudo apt install python3-pip
    ```

2.  **Clone the MAVLink Repository:**
    ```bash
    git clone https://github.com/mavlink/mavlink.git --recursive
    cd mavlink
    ```

3.  **Install Python Requirements:**
    ```bash
    python3 -m pip install -r pymavlink/requirements.txt
    ```

4.  **Generate C Headers (using `common.xml`):**
    ```bash
    python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
    ```

5.  **Using with CMake:** Install to a local directory (e.g., `install`):
    ```bash
    cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
    cmake --build build --target install
    ```

6.  **Integrate in CMake:**  Use `find_package` in your `CMakeLists.txt`:
    ```cmake
    find_package(MAVLink REQUIRED)
    add_executable(my_program my_program.c)
    target_link_libraries(my_program PRIVATE MAVLink::mavlink)
    ```

7.  **Build your program:**  Pass the install directory to CMake:
    ```bash
    cd ../my_program
    cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
    ```

For more detailed C library instructions, see [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/).

## Further Resources

*   **Documentation/Website:** [https://mavlink.io/en/](https://mavlink.io/en/)
*   **Discussion/Support:** [https://mavlink.io/en/#support](https://mavlink.io/en/#support)
*   **Contributing:** [https://mavlink.io/en/contributing/contributing.html](https://mavlink.io/en/contributing/contributing.html)
*   **License:** [https://mavlink.io/en/#license](https://mavlink.io/en/#license)
*   **Installation Guides:** [Installing the MAVLink Toolchain](https://mavlink.io/en/getting_started/installation.html)
*   **Library Generation:** [Generating MAVLink Libraries](https://mavlink.io/en/getting_started/generate_libraries.html)
*   **Using Libraries:** [Using MAVLink Libraries](https://mavlink.io/en/getting_started/use_libraries.html)