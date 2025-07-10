# NUR: The Nix User Repository

**Expand your Nix package ecosystem with the Nix User Repository (NUR), a community-driven collection of user-contributed Nix packages.**

[![GitHub Repo stars](https://img.shields.io/github/stars/nix-community/NUR?style=social)](https://github.com/nix-community/NUR)

NUR allows you to quickly access and install Nix packages from the community, built from source and independently maintained. This contrasts with the curated packages of Nixpkgs, offering a faster, more decentralized approach to package discovery and availability.

**Key Features:**

*   **Community-Driven:** Access a wide range of packages contributed by Nix users.
*   **Decentralized Package Hosting:**  Packages are built from source, hosted in user repositories, and not subject to Nixpkgs review.
*   **Automated Evaluation Checks:** NUR automatically validates repositories before updating its package list, reducing the risk of broken packages.
*   **Easy Integration:**  Integrate NUR packages with your existing Nix configurations using flakes, package overrides, NixOS modules, and Home Manager.
*   **Package Discovery:** Find packages through the [NUR package search](https://nur.nix-community.org/) or the [nur-combined](https://github.com/nix-community/nur-combined) repository.
*   **Flexible Repository Management:** Add your own packages and repositories to the NUR ecosystem.
*   **Overriding Repositories:** Test changes before publishing.

**Installation:**

NUR can be integrated into your Nix environment through various methods, including:

*   **Using Flakes:** (See original README for code example)
*   **Using `packageOverrides`:** (See original README for code example)

**How to Use:**

Once installed, you can easily install packages from the NUR namespace using commands like `nix-shell` and `nix-env`, or by incorporating them into your `configuration.nix` file.  (See original README for example commands).

**Adding Your Own Repository:**

1.  Create a Git repository with a `default.nix` file at its root, defining your packages.  Consider using the [repository template](https://github.com/nix-community/nur-packages-template).
2.  Follow the guidelines for structuring your `default.nix` and incorporating packages from Nixpkgs.
3.  Edit `repos.json` in the NUR repository (after cloning the NUR repo - see instructions in the original README).
4.  Add your repository details and submit a pull request to the main NUR repository.
5.  Update the NUR lock file using the [nur-update service](https://nur-update.nix-community.org/)

**Important Considerations:**

*   **Security:** NUR does not perform regular security audits.  Always review package expressions before installation.
*   **Package Quality:**  Packages are not reviewed by Nixpkgs maintainers.
*   **Evaluation Errors:**  Ensure your packages build successfully and adhere to NUR's guidelines to avoid evaluation errors.

**Contributing:**

*   Follow the contribution guidelines in the README for adding packages.
*   Set the `meta.broken` attribute to `true` if a package is not building.
*   Utilize Nixpkgs packages when possible.
*   Provide metadata as outlined in the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).

**Community & Support:**

Join the NUR community for discussions and support:

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org](https://discourse.nixos.org/)

[Back to Top](#top)

---

**Learn more and contribute at the original repository: [github.com/nix-community/NUR](https://github.com/nix-community/NUR)**