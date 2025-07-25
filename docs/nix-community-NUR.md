# Nix User Repository (NUR): Extend Your Nix Ecosystem

**NUR empowers the Nix community by providing a decentralized platform for sharing and installing user-contributed packages, modules, and overlays.** [Visit the original repository](https://github.com/nix-community/NUR)

## Key Features

*   **Community-Driven:** Access a wide array of packages, modules, and overlays created and maintained by the Nix community.
*   **Decentralized:** Discover and install packages from user repositories without the need for centralized review (use with caution - see security notes).
*   **Easy Integration:** Seamlessly integrate NUR with your Nix configurations using flakes, `packageOverrides`, and Home Manager.
*   **Automated Updates:** NUR automatically checks and evaluates repositories for updates, ensuring the stability of your system.
*   **Flexible Package Discovery:** Find packages through a dedicated search interface and the `nur-combined` repository.
*   **Support for NixOS Modules, Overlays and Library Functions:** Use the repository for all kind of Nix expressions.

## Installation

### Using Flakes

Add NUR to your `flake.nix`:

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nur = {
      url = "github:nix-community/NUR";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
};
```

### Using `packageOverrides`

Add the following to `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add this to `/etc/nixos/configuration.nix`:

```nix
{
  nixpkgs.config.packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

### Pinning

Pin the version for faster builds and offline access:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages using:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

or

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Security Note:***  NUR packages are not reviewed by the Nixpkgs maintainers.  Always review the source of packages before installation.

## Finding Packages

*   [Packages search for NUR](https://nur.nix-community.org/)
*   [nur-combined](https://github.com/nix-community/nur-combined/search)

## How to Add Your Own Repository

1.  Create a repository with a `default.nix`.
2.  Follow the repository structure from the [template](https://github.com/nix-community/nur-packages-template).
3.  Add your repository details to NUR's `repos.json`.
4.  Open a pull request to the NUR repository.

## Overriding Repositories

Override repositories using `repoOverrides` to test changes:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
      repoOverrides = {
        mic92 = import ../nur-packages { inherit pkgs; };
        ## remote locations are also possible:
        # mic92 = import (builtins.fetchTarball "https://github.com/your-user/nur-packages/archive/main.tar.gz") { inherit pkgs; };
      };
    };
  };
}
```

## Contribution Guidelines

*   Ensure packages build and set `meta.broken` if necessary.
*   Provide [Nixpkgs-compliant meta attributes](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories lean and reuse Nixpkgs packages.

## Contact

Join the conversation on [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org) and [Discourse](https://discourse.nixos.org/).