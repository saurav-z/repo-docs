<!-- START of README -->
# NUR: The Nix User Repository

**NUR empowers Nix users to share and install community-contributed packages quickly and easily.**

Nix User Repository (NUR) is a community-driven meta-repository that extends Nix's package management capabilities. It provides access to user-maintained repositories containing Nix package descriptions, allowing you to install packages not available in [Nixpkgs](https://github.com/NixOS/nixpkgs/) directly. Packages are built from source and are not reviewed by Nixpkgs members.

## Key Features

*   **Community-Driven:** Access packages contributed and maintained by the Nix community.
*   **Decentralized:** Share and discover packages faster than traditional package repositories.
*   **Flexible:** Install packages via attribute references.
*   **Automated Evaluation:** NUR automatically checks repositories and performs evaluation checks to ensure package integrity.
*   **Easy Integration:** Seamlessly integrates with Nix flakes, `packageOverrides`, and Home Manager.

## Installation

### Using Flakes

Include NUR in your `flake.nix`:

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

Then, either the overlay (`overlays.default`) or `legacyPackages.<system>` can be used.

### Using `packageOverrides`

Add the following to your `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add the following to your `/etc/nixos/configuration.nix`:

```nix
{
  nixpkgs.config.packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

### Pinning (Recommended for Stability)

Pinning ensures consistent builds by specifying a specific NUR revision:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

After installation, packages are available within the NUR namespace.

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

**Important:** *NUR does not regularly check repositories for malicious content. Always review packages before installation.*

### Example: Single Package in a Devshell (Flake)

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
          overlays = [ nur.overlays.default ];
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

### Using NUR in NixOS (Modules and Overlays)

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
      ];
    };
  };
}
```

### Integrating with Home Manager

Integrate with Home Manager by importing modules into the `imports` attribute.

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

*   **Search:** Browse packages at [Packages search for NUR](https://nur.nix-community.org/).
*   **Browse:** Explore the [nur-combined](https://github.com/nix-community/nur-combined) repository on GitHub.

## Adding Your Repository

1.  **Create a Repository:** Create a Git repository with a `default.nix` file in the root, which defines your packages. Use the [repository template](https://github.com/nix-community/nur-packages-template) for structure.
2.  **Package Definition:**  Packages should return a set of Nix derivations, and get dependencies from the `pkgs` argument.
3.  **Example Package (`hello-nur`):**

    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```

    Where `hello-nur/default.nix` might be:

    ```nix
    { stdenv, fetchurl, lib }:
    stdenv.mkDerivation rec {
      # ... (package definition) ...
    };
    ```

4.  **Testing:** Use `nix-shell --arg pkgs 'import <nixpkgs> {}' -A hello-nur` or `nix-build --arg pkgs 'import <nixpkgs> {}' -A hello-nur` to test your packages.

5.  **Add to `repos.json`:**  Edit the `repos.json` file in the NUR repository (clone first):

    ```bash
    git clone --depth 1 https://github.com/nix-community/NUR
    cd NUR
    ```

    Edit `repos.json`:

    ```json
    {
        "repos": {
            "mic92": {
                "url": "https://github.com/Mic92/nur-packages"
            },
            "<fill-your-repo-name>": {
                "url": "https://github.com/<your-user>/<your-repo>"
            }
        }
    }
    ```

6.  **Format and Commit:**

    ```bash
    ./bin/nur format-manifest
    git add repos.json
    git commit -m "add <your-repo-name> repository"
    git push
    ```

7.  **Create a Pull Request:** Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

**Notes:**

*   Repositories should be buildable on Nixpkgs unstable.
*   To use a different root file, use the `file` option in `repos.json`.
*   To fetch git submodules set `submodules` to `true` in `repos.json`.

### Updating Your Repository (Fast)

After updating your repository, use the [nur-update service](https://nur-update.nix-community.org/) to trigger a faster update of the NUR's lock file.

```bash
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Troubleshooting Package Updates

*   **Evaluation Errors:** Check the [latest build job](https://github.com/nix-community/NUR/actions) for evaluation errors. Common issues include incorrect licenses or using fetchers that access external URLs during evaluation.
*   **Local Evaluation:** Use the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task in your repository for local checking.

### Overriding Repositories

You can override repositories using the `repoOverrides` argument.  This is useful for testing changes before publishing.  There are examples for use with both `packageOverrides` and with Nix Flakes.

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if not.
*   Provide standard `meta` attributes.
*   Keep repositories lean.
*   Reuse packages from Nixpkgs whenever possible.

## Potential NUR Package Use Cases

*   Packages for a small audience.
*   Pre-releases.
*   Older package versions.
*   Automatically generated package sets.
*   Software with custom patches.
*   Experiments.

## Support and Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)

<!-- END of README -->