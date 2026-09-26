{
  description = "NInfer development environment (CUDA 13.1, Python 3.13)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";

      # The CUDA toolkit is unfree (CUDA EULA).
      pkgs = (import nixpkgs) {
        inherit system;
        config.allowUnfree = true;
      };

      cuda = pkgs.cudaPackages_13_1;
      python = pkgs.python313.withPackages (
        ps: with ps; [
          jinja2
          numpy
          pytest
          pyyaml
          rich
          safetensors
          torch
        ]
      );
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "ninfer-dev";
        packages = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
          cuda.cuda_nvcc
          cuda.cuda_cudart
          cuda.cuda_nvtx
          cuda.nsight_compute
          cuda.nsight_systems
          pkgs.ffmpeg
          pkgs.curl
          python
        ];
      };
    };
}
