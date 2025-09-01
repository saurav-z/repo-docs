# GDSFactory: Design Chips with Python - Photonic, Analog, Quantum, and More

**GDSFactory is a powerful Python library enabling you to design chips for photonics, analog, quantum, and more, streamlining your hardware design process and making it accessible to all.**  You can find the original repository [here](https://github.com/gdsfactory/gdsfactory).

**Key Features:**

*   **Parametric Design:** Define and generate complex chip components using Python code.
*   **Multi-Format Output:** Produce industry-standard CAD files, including GDSII, OASIS, STL, and GERBER.
*   **Integrated Simulation:** Seamlessly integrate with popular simulation tools for design verification and optimization.
*   **Verification & Validation:** Benefit from built-in DRC (Design Rule Checking), DFM (Design for Manufacturing), and LVS (Layout Versus Schematic) capabilities, plus automated chip analysis.
*   **Open-Source Ecosystem:** Leverage the power of a vibrant and growing community, with numerous open-source PDKs (Process Design Kits) available.
*   **Extensible and Flexible:** Easily add custom components and extend functionality to meet your specific design needs.
*   **Community Support:** Connect with other users and developers through GitHub Discussions, Google Groups, and LinkedIn.
*   **Fast Performance:** Experience the speed of KLayout C++ library for manipulating GDS objects.

**Benefits:**

*   **Accelerated Design:** Significantly reduce design time with Python-based parametric design.
*   **Enhanced Accuracy:** Minimize errors with integrated verification tools.
*   **Simplified Workflow:** Create an end-to-end chip design flow, from design to validation.
*   **Open and Accessible:** Benefit from a free and open-source library that promotes collaboration.

**Getting Started:**

1.  **Installation:**
    ```bash
    pip install gdsfactory_install
    gfi install
    ```
2.  **Example:**
    ```python
    import gdsfactory as gf

    # Create a new component
    c = gf.Component()

    # Add a rectangle
    r = gf.components.rectangle(size=(10, 10), layer=(1, 0))
    rect = c.add_ref(r)

    # Add text elements
    t1 = gf.components.text("Hello", size=10, layer=(2, 0))
    t2 = gf.components.text("world", size=10, layer=(2, 0))

    text1 = c.add_ref(t1)
    text2 = c.add_ref(t2)

    # Position elements
    text1.xmin = rect.xmax + 5
    text2.xmin = text1.xmax + 2
    text2.rotate(30)

    # Show the result
    c.show()
    ```

**Key Metrics:**

*   **+2M Downloads**
*   **+81 Contributors**
*   **+25 PDKs Available**

**PDKs:**

*   **Open-Source PDKs (No NDA Required):**
    *   GlobalFoundries 180nm MCU CMOS PDK
    *   ANT / SiEPIC Ebeam UBC PDK
    *   SkyWater 130nm CMOS PDK
    *   VTT PDK
    *   Cornerstone PDK
    *   Luxtelligence GF PDK
*   **Foundry PDKs (NDA Required - GDSFactory+ Subscription):**
    *   AIM Photonics
    *   AMF Photonics
    *   CompoundTek Photonics
    *   Fraunhofer HHI Photonics
    *   Smart Photonics
    *   Tower Semiconductor PH18
    *   Tower PH18DA by OpenLight
    *   III-V Labs
    *   LioniX
    *   Ligentec
    *   Lightium
    *   Quantum Computing Inc. (QCI)

**Resources:**

*   [Documentation](https://gdsfactory.github.io/gdsfactory/)
*   [Video Tutorials](https://www.youtube.com/@gdsfactory/playlists)
*   [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions)
*   [Google Group](https://groups.google.com/g/gdsfactory)
*   [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [Slack community channel](https://join.slack.com/t/gdsfactory-community/shared_invite/zt-3aoygv7cg-r5BH6yvL4YlHfY8~UXp0Wg)
*   [GDSFactory Website](https://gdsfactory.com)