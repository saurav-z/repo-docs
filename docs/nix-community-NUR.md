# Nix User Repository (NUR): Community-Driven Package Sharing for Nix

NUR is a powerful community-driven meta-repository that extends the Nix package ecosystem, offering a decentralized way to access user-contributed packages. **Expand your Nix package options with NUR, the community hub for innovative software and experimental builds.**

**Key Features:**

*   **Community-Driven:** Access a wide array of packages contributed and maintained by the Nix community.
*   **Decentralized Package Sharing:** Easily share and install packages not available in Nixpkgs.
*   **Rapid Package Discovery:** Obtain new packages quickly and without the formal review process of Nixpkgs.
*   **Automated Checks:** Benefit from evaluation checks and updates to maintain package integrity.
*   **Flake and Overlays Support:** Integrates seamlessly with modern Nix workflows.
*   **Extensible:** Supports NixOS modules, overlays, and library functions.
*   **Flexible Installation:** Install packages via `nix-shell`, `nix-env`, or `configuration.nix`.

**Installation**

Choose your preferred method:

*   **Using Flakes:** Integrate NUR into your `flake.nix` file (see the original README for detailed code).
*   **Using `packageOverrides`:** Add NUR to `~/.config/nixpkgs/config.nix` or your `/etc/nixos/configuration.nix` (see the original README for detailed code).

**How to Use**

Once installed, access packages from NUR by referencing them within the `nur.repos.<user>.<package>` namespace, such as:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
```

Or:

```bash
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

Or:

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important Security Note:**  *Always* review the expressions before installing packages from NUR.  While the community strives for quality, packages are not subject to Nixpkgs-level review.

**Finding Packages:**

*   **Package Search:** [Packages search for NUR](https://nur.nix-community.org/)
*   **Combined Repository:** Search the [nur-combined](https://github.com/nix-community/nur-combined) repository on GitHub.

**Adding Your Repository:**

1.  Create a Git repository with a `default.nix` file (consider using the [repository template](https://github.com/nix-community/nur-packages-template)).
2.  Use `pkgs` argument, and do NOT import packages directly from `<nixpkgs>`.
3.  In your `repos.json`, add a new entry that references your git repo.
4.  Use `bin/nur format-manifest` to sort, and commit, and then create a pull request to the NUR repository.

**Additional Capabilities:**

*   **Git Submodules:** Enable submodules in your repository configuration.
*   **NixOS Modules, Overlays & Library Functions:** Organize modules, overlays, and library functions for easy discovery.
*   **Overriding Repositories:** Test changes before publishing (using `repoOverrides`).
*   **Flake-Based Overrides:** Experimental support for repository overrides within a Flake context.

**Contribution Guidelines**

*   Ensure packages build and set `meta.broken = true` if they don't.
*   Supply metadata attributes, per the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories concise.
*   Reuse Nixpkgs packages when appropriate.

**Get Help:**

*   Matrix channel: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org](https://discourse.nixos.org/)

[View the original repository](https://github.com/nix-community/NUR)