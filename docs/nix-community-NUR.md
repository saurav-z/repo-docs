# NUR: Your Community Hub for Nix Packages

**Nix User Repository (NUR) empowers Nix users to discover, share, and install community-contributed packages, expanding the Nix ecosystem.** [Explore the NUR on GitHub](https://github.com/nix-community/NUR)

## Key Features

*   **Community-Driven:** Access a wide variety of user-submitted Nix packages.
*   **Decentralized and Rapid Updates:** Get access to new packages faster than traditional Nixpkgs.
*   **Easy Installation:** Integrate NUR with flakes, packageOverrides, NixOS configurations, and Home Manager.
*   **Automated Checks:** Benefit from automated evaluation checks to ensure package integrity.
*   **Flexible Usage:** Integrate NUR into your devshells, NixOS configurations, and Home Manager setups with ease.
*   **Package Discovery:** Find packages through the NUR package search or via `nur-combined`.

## Installation

### Using Flakes

Integrate NUR into your `flake.nix`:

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

Then, use either `overlays.default` or `legacyPackages.<system>`.

### Using `packageOverrides`

Add this to `~/.config/nixpkgs/config.nix`:

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

### Pinning

Pinning is recommended to ensure build reproducibility:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Use packages from NUR:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
```

or

```bash
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or in your configuration:

```nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Important Security Note:**  It is recommended to review package expressions before installing them, as NUR is community-driven and packages are not reviewed by Nixpkgs members.*

### Devshell Example

Add a single package from NUR to a devshell:

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

### NixOS Integration

Using overlays and modules from NUR in your configuration:

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

### Home Manager Integration

Integrate with Home Manager using `imports`:

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

## Adding Your Repository

1.  Create a repository with a `default.nix` file.  A [repository template](https://github.com/nix-community/nur-packages-template) is available.
2.  Use `pkgs` as the argument for dependencies from Nixpkgs.
3.  Return a set of Nix derivations.

```nix
{ pkgs }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

Example `hello-nur/default.nix`:

```nix
{ stdenv, fetchurl, lib }:

stdenv.mkDerivation rec {
  pname = "hello";
  version = "2.10";

  src = fetchurl {
    url = "mirror://gnu/hello/${pname}-${version}.tar.gz";
    sha256 = "0ssi1wpaf7plaswqqjwigppsg5fyh99vdlb9kzl7c9lng89ndq1i";
  };

  postPatch = ''
    sed -i -e 's/Hello, world!/Hello, NUR!/' src/hello.c
  '';

  # fails due to patch
  doCheck = false;

  meta = with lib; {
    description = "A program that produces a familiar, friendly greeting";
    longDescription = ''
      GNU Hello is a program that prints "Hello, world!" when you run it.
      It is fully customizable.
    '';
    homepage = https://www.gnu.org/software/hello/manual/;
    changelog = "https://git.savannah.gnu.org/cgit/hello.git/plain/NEWS?h=v${version}";
    license = licenses.gpl3Plus;
    maintainers = [ maintainers.eelco ];
    platforms = platforms.all;
  };
}
```

Build your package:

```bash
$ nix-shell --arg pkgs 'import <nixpkgs> {}' -A hello-nur
```

Add your repository to `repos.json`:

```bash
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

Run these commands:

```bash
$ ./bin/nur format-manifest # ensure repos.json is sorted alphabetically
$ git add repos.json
$ git commit -m "add <your-repo-name> repository"
$ git push
```

Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Using a Different Root Expression

To use a different file other than `default.nix`:

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

### Update NUR's Lock File

Update NUR faster after your repository update via the `nur-update` service:

```bash
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Troubleshooting Package Updates

*   Verify your package's evaluation succeeds by checking the [latest build job](https://github.com/nix-community/NUR/actions).
*   Check for common evaluation errors as detailed in the original README.
*   Use the Local evaluation check from the original README to troubleshoot locally.

### Git Submodules

To fetch git submodules, set `submodules`:

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

### NixOS Modules, Overlays, and Library Function Support

*   **NixOS Modules:** Place modules in the `modules` attribute.
*   **Overlays:** Use the `overlays` attribute.
*   **Library Functions:** Place reusable functions in the `lib` attribute.

## Overriding Repositories

You can override repositories with the `repoOverrides` argument.

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

### Flake-based Overrides (Experimental)

- **packageOverrides**:
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

- **overlay**:
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

The repo must contain a `flake.nix` file.

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if not.
*   Supply meta attributes per the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories slim.
*   Reuse packages from Nixpkgs when possible.

## Potential NUR Use Cases

*   Packages for a small audience.
*   Pre-releases.
*   Old package versions.
*   Automatically generated package sets.
*   Software with opinionated patches.
*   Experiments.

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org](https://discourse.nixos.org/)