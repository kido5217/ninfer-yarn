#!/usr/bin/env python3.13
"""Independent YaRN RoPE table construction reference (torch, fp64).

This is the independent oracle for the *table construction* half of the YaRN RoPE
extension: it computes the per-pair inverse frequencies (``inv_freq``) and the
attention magnitude (``mscale``) from the spec formulas

    corr_dim(n_rot) = R * ln(L / (n_rot * 2*pi)) / (2 * ln(base))
    low  = floor(corr_dim(beta_fast))          clamped to [0, R-1]
    high = ceil (corr_dim(beta_slow))          clamped to [0, R-1]
    ramp(i)  = clamp((i - low) / (high - low), 0, 1)     i = pair index 0..R/2-1
    mask(i)  = (1 - ramp(i)) * ext_factor
    extrap[i] = base ** (-2i / R)
    interp[i] = extrap[i] / s
    inv_freq[i] = interp[i] * (1 - mask[i]) + extrap[i] * mask[i]
    mscale     = 1.0 if s <= 1 else 0.1 * ln(s) + 1

The output is the source of the golden vectors committed into
``tests/ops/test_rope.cpp`` (the C++ suite asserts the production host helper
reproduces these values). It is deliberately a standalone torch fp64
implementation, independent of the C++ production code path.

Run from the repository root with the devShell Python (Python 3.13, torch):

    nix develop -c python3.13 tools/yarn_table_check.py
"""

from __future__ import annotations

import argparse
import math
import sys

import torch


def corr_dim(num_rotations: float, dim: int, base: float, max_position: int) -> float:
    return (
        dim
        * math.log(max_position / (num_rotations * 2.0 * math.pi))
        / (2.0 * math.log(base))
    )


def find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float,
    max_position: int,
) -> tuple[int, int]:
    low = math.floor(corr_dim(low_rot, dim, base, max_position))
    high = math.ceil(corr_dim(high_rot, dim, base, max_position))
    return max(low, 0), min(high, dim - 1)


def build_table(
    base: float,
    rotary_dim: int,
    original_max: int,
    scale: float,
    beta_fast: int = 32,
    beta_slow: int = 1,
    ext_factor: float = 1.0,
) -> dict:
    """Compute the YaRN table in torch fp64 for the pinned geometry."""
    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")
    pairs = rotary_dim // 2
    low, high = find_correction_range(
        beta_fast, beta_slow, rotary_dim, base, original_max
    )
    if low == high:
        high += 0.001  # avoid a zero-width ramp, mirroring the reference
    # inv_freq over the pair index, fp64.
    idx = torch.arange(pairs, dtype=torch.float64)
    ramp = torch.clamp((idx - float(low)) / (float(high) - float(low)), 0.0, 1.0)
    mask = (1.0 - ramp) * ext_factor
    # Base (pure power-law) frequency for pair index i: theta ** (-2i / rotary_dim).
    # This matches the production `kTextRopeInvFrequency` `__constant__` table, so the
    # s == 1.0 table is byte-identical to the non-YaRN path.
    extrap = torch.pow(torch.tensor(base, dtype=torch.float64), -2.0 * idx / rotary_dim)
    interp = extrap / scale
    inv_freq = interp * (1.0 - mask) + extrap * mask
    if scale <= 1.0:
        mscale = 1.0
    else:
        mscale = 0.1 * math.log(scale) + 1.0
    return {
        "low": low,
        "high": high,
        "mscale": mscale,
        "inv_freq": inv_freq.tolist(),
    }


def _fmt_float(value: float) -> str:
    # 17 significant digits: exact round-trip for a double.
    return repr(float(value))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base", type=float, default=1.0e7, help="RoPE theta base (default 1e7)"
    )
    parser.add_argument(
        "--rotary-dim", type=int, default=64, help="rotary dimension (default 64)"
    )
    parser.add_argument(
        "--original-max", type=int, default=262144, help="original context L"
    )
    parser.add_argument("--scale", type=float, default=2.0, help="extension factor s")
    parser.add_argument("--beta-fast", type=int, default=32)
    parser.add_argument("--beta-slow", type=int, default=1)
    parser.add_argument("--ext-factor", type=float, default=1.0)
    parser.add_argument(
        "--c",
        action="store_true",
        help="emit C++ array initializers (inv_freq + mscale) for the committed golden vector",
    )
    args = parser.parse_args()

    table = build_table(
        args.base,
        args.rotary_dim,
        args.original_max,
        args.scale,
        args.beta_fast,
        args.beta_slow,
        args.ext_factor,
    )
    pairs = args.rotary_dim // 2

    if args.c:
        print(f"low={table['low']} high={table['high']} scale={args.scale}")
        print(f"static constexpr double kGoldenMscale = {_fmt_float(table['mscale'])}")
        print(f"static constexpr double kGoldenInvFreq[{pairs}] = {{")
        for i, value in enumerate(table["inv_freq"]):
            suffix = "," if i < pairs - 1 else ""
            print(f"    {_fmt_float(value)}{suffix}")
        print("};")
    else:
        print(
            f"base={args.base} rotary_dim={args.rotary_dim} L={args.original_max} s={args.scale}"
        )
        print(
            f"beta_fast={args.beta_fast} beta_slow={args.beta_slow} ext_factor={args.ext_factor}"
        )
        print(f"low={table['low']} high={table['high']} mscale={table['mscale']!r}")
        print("inv_freq:")
        for i, value in enumerate(table["inv_freq"]):
            print(f"  [{i:2d}] {value!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
