# NUR: The Nix User Repository

**Expand your Nix package ecosystem with the Nix User Repository (NUR), a community-driven hub for user-contributed Nix packages.**

[Link to Original Repo](https://github.com/nix-community/NUR)

NUR allows you to easily access and install packages created and maintained by the Nix community, providing a faster and more decentralized way to share and use new packages. Unlike Nixpkgs, packages in NUR are built from source and are not reviewed by Nixpkgs members, offering a broader range of software options.

**Key Features:**

*   **Community-Driven:** Access a wide range of user-contributed packages.
*   **Decentralized:** Easily share and use packages without going through a central review process.
*   **Automated Updates:** NUR automatically checks repositories and performs evaluation checks before propagating updates.
*   **Flexible Installation:** Integrate NUR with flakes, package overrides, and NixOS configurations.
*   **Package Discovery:** Find packages through the dedicated NUR search or the combined repository.
*   **NixOS Module, Overlay, and Library Support:** Extend your NixOS and Home Manager configurations with modules, overlays, and library functions from NUR.
*   **Easy Contribution:** Add your own packages and repositories to the NUR ecosystem.

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

Use the overlay (`overlays.default`) or `legacyPackages.<system>`.

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

For NixOS add the following to your `/etc/nixos/configuration.nix`
Notice: If you want to use NUR in nix-env, home-manager or in nix-shell you also need NUR in `~/.config/nixpkgs/config.nix`
as shown above!

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

To avoid internet access during builds, pin the NUR version:

```nix
builtins.fetchTarball {
  # Get the revision by choosing a version from https://github.com/nix-community/NUR/commits/main
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  # Get the hash by running `nix-prefetch-url --unpack <url>` on the above url
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install and use packages from the NUR namespace.

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

```console
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important Note:** _It is recommended to review packages before installing them._

## Examples

### Using a Single Package in a Devshell

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

### Using the Flake in NixOS

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
*   [nur-combined](https://github.com/nix-community/nur-combined/search)

## Contributing

### How to Add Your Own Repository

1.  Create a repository with a `default.nix` file. Use the [repository template](https://github.com/nix-community/nur-packages-template).
2.  Do **NOT** import packages using `with import <nixpkgs> {};`. Instead, use the provided `pkgs` argument.
3.  Each repository should return a set of Nix derivations.
    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```
4.  Add your repository to `repos.json`.

    ```console
    $ git clone --depth 1 https://github.com/nix-community/NUR
    $ cd NUR
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

    Run `bin/nur update` and commit `repos.json` (but **NOT** `repos.json.lock`).

    ```
    $ ./bin/nur format-manifest # ensure repos.json is sorted alphabetically
    $ git add repos.json
    $ git commit -m "add <your-repo-name> repository"
    $ git push
    ```

    Open a pull request at [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

###  Using a Different Nix File as Root Expression

Set the `file` option in `repos.json`:

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages",
            "file": "subdirectory/default.nix"
        }
    }
}
```

###  Updating NUR's Lock File

Use [nur-update](https://nur-update.nix-community.org/) after pushing updates:

```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

###  Why are NUR Packages Not Updating?

Check the [latest build job](https://github.com/nix-community/NUR/actions) for evaluation errors.

###  Local Evaluation Check

In your `nur-packages/` directory, run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task.

###  Git Submodules

Set `submodules`:

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages",
            "submodules": true
        }
    }
}
```

### NixOS Modules, Overlays, and Library Functions

*   **NixOS Modules:** Put in the `modules` attribute.
    ```nix
    { pkgs }: {
      modules = import ./modules;
    }
    ```
    ```nix
    # modules/default.nix
    {
      example-module = ./example-module.nix;
    }
    ```
    Modules should be defined as paths and the `_class` attribute set to `"nixos"` or [`"home-manager"`](https://github.com/nix-community/home-manager/commit/26e72d85e6fbda36bf2266f1447215501ec376fd) if NixOS or Home Manager specific.
*   **Overlays:** Use the `overlays` attribute.
    ```nix
    # default.nix
    {
      overlays = {
        hello-overlay = import ./hello-overlay;
      };
    }
    ```
    ```nix
    # hello-overlay/default.nix
    self: super: {
      hello = super.hello.overrideAttrs (old: {
        separateDebugInfo = true;
      });
    }
    ```
*   **Library Functions:** Put in the `lib` attribute.
    ```nix
    { pkgs }:
    with pkgs.lib;
    {
      lib = {
        hexint = x: hexvals.${toLower x};

        hexvals = listToAttrs (imap (i: c: { name = c; value = i - 1; })
          (stringToCharacters "0123456789abcdef"));
      };
    }
    ```

## Overriding Repositories

###  `repoOverrides` Argument
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

###  With Flakes (Experimental)
```nix
{
  inputs.nur.url = "github:nix-community/NUR";
  inputs.paul.url = "path:/some_path/nur-paul"; # example: a local nur.repos.paul for development

  outputs = {self, nixpkgs, nur, paul }: {

  system = "x86_64-linux";

  nurpkgs = import nixpkgs { inherit system; };

  ...
  modules = [
       {
         nixpkgs.config.packageOverrides = pkgs: {
            nur = import nur {
              inherit pkgs nurpkgs;
              repoOverrides = { paul = import paul { inherit pkgs; }; };
            };
          };
        }
  ];
  ...
}
```

or
```nix
{
  modules = [
    {
      nixpkgs.overlays = [
        (final: prev: {
          nur = import nur {
            nurpkgs = prev;
            pkgs = prev;
            repoOverrides = { paul = import paul { pkgs = prev; }; };
          };
        })
      ];
    }
    ...
  ];
}
```

The repo must contain a `flake.nix` file in addition to a `default.nix`:  [flake.nix example](https://github.com/Mic92/nur-packages/blob/master/flake.nix)

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if they don't.
*   Provide standard Nixpkgs meta attributes.
*   Keep repositories slim.
*   Reuse packages from Nixpkgs when applicable.

**Examples of suitable packages:**

*   Packages for a niche audience
*   Pre-releases of software
*   Old versions of packages
*   Automatically generated package sets
*   Software with opinionated patches
*   Experiments

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)