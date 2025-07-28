# NUR: The Nix User Repository

**Expand your Nix package options with the community-driven Nix User Repository (NUR)!**

[Link to Original Repo](https://github.com/nix-community/NUR)

NUR is a community-driven meta-repository for Nix packages, offering a decentralized way to access and install user-contributed packages. Unlike Nixpkgs, packages in NUR are built from source and not subject to the same review process. This allows for faster sharing of new and experimental packages, pre-releases, and software tailored to specific needs.

## Key Features

*   **Community-Driven:** Access packages contributed by the Nix community.
*   **Decentralized:** Easily share and install packages without going through Nixpkgs review.
*   **Fast Updates:** Get access to new packages and updates quickly.
*   **Flexible:** Suitable for a wide range of packages, including pre-releases, specialized software, and custom builds.
*   **Automatic Checks:** NUR performs evaluation checks before propagating updates.

## Installation

### Using Flakes

Integrate NUR into your `flake.nix` to utilize the overlay (`overlays.default`) or `legacyPackages.<system>`:

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

Add NUR to your `packageOverrides` for your login user by adding this to `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add the following to `/etc/nixos/configuration.nix`.  **Important:** If you plan to use NUR in `nix-env`, `home-manager`, or `nix-shell`, also include it in `~/.config/nixpkgs/config.nix`.

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

To avoid frequent downloads and speed up builds, pin your NUR version:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Once installed, you can use or install packages from the NUR namespace.

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
```

or

```bash
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Important: Review package expressions before installing them due to the community-driven nature of NUR.***

### Devshell Example

Add a single package from NUR to a devshell within your `flake.nix`:

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nur = {
      url = "github:nix-community/NUR";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, nur }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nur.overlay ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.nur.repos.mic92.hello-nur ];
        };
      }
    );
}
```

### NixOS Configuration

Easily integrate NUR modules and overlays into your NixOS configuration:

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nur = {
      url = "github:nix-community/NUR";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, nur }: {
    nixosConfigurations.myConfig = nixpkgs.lib.nixosSystem {
      # ...
      modules = [
        # Adds the NUR overlay
        nur.modules.nixos.default
        # NUR modules to import
        nur.legacyPackages."${system}".repos.iopq.modules.xraya
        # This adds the NUR nixpkgs overlay.
        # Example:
        # ({ pkgs, ... }: {
        #   environment.systemPackages = [ pkgs.nur.repos.mic92.hello-nur ];
        # })
      ];
    };
  };
}
```

### Integrating with Home Manager

Integrate NUR modules into your Home Manager configuration:

```nix
let
  nur-no-pkgs = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {};
in
{
  imports = lib.attrValues nur-no-pkgs.repos.moredhel.hmModules.rawModules;

  services.unison = {
    enable = true;
    profiles = {
      org = {
        src = "/home/moredhel/org";
        dest = "/home/moredhel/org.backup";
        extraArgs = "-batch -watch -ui text -repeat 60 -fat";
      };
    };
  };
}
```

## Finding Packages

*   [Packages search for NUR](https://nur.nix-community.org/)
*   [nur-combined](https://github.com/nix-community/nur-combined)

## Adding Your Own Repository

1.  **Create a Repository:**  Start with a `default.nix` file in the root directory.  A [repository template](https://github.com/nix-community/nur-packages-template) is available.
2.  **Dependencies:**  Use the `pkgs` argument to import dependencies from Nixpkgs (e.g., `pkgs.hello`).
3.  **Package Definitions:** Each repository should return a set of Nix derivations.
4.  **Build and Test:** Use `nix-shell` or `nix-build` to build and test your packages.
5.  **Add to `repos.json`:** Add your repository's URL to the `repos.json` file in the NUR repository.
6.  **Submit a Pull Request:**  Create a pull request to the [NUR GitHub repository](https://github.com/nix-community/NUR).

### Advanced Configuration

*   **Different Root File:** Use the `file` option in `repos.json` to load packages from a different file than `default.nix`.
*   **Updating the Lockfile:** Use the [nur-update service](https://nur-update.nix-community.org/update?repo=<your_repo_name>) to speed up updates after changes.
*   **Troubleshooting Updates:**  Check the [latest build job](https://github.com/nix-community/NUR/actions) for evaluation errors.
*   **Git Submodules:** Enable Git submodule support with the `submodules` option.
*   **NixOS Modules, Overlays, and Library Functions:** Structure your repository with `modules`, `overlays`, and `lib` attributes to provide more than just packages.
*   **Overriding Repositories:** Use `repoOverrides` in `packageOverrides` or with Flakes for testing changes locally before publishing.

## Contribution Guidelines

*   Ensure packages build successfully and set `meta.broken = true` if they do not.
*   Provide standard meta attributes (as described in the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes)).
*   Keep repositories lean, utilizing Nixpkgs packages where possible.

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org](https://discourse.nixos.org/)