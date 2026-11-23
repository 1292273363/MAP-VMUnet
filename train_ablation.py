# SPDX-License-Identifier: Apache-2.0
"""Run the MDK/PARD component ablations or the PARD internal ablations."""

from map_vmunet.runner import load_config, run, run_parser


COMPONENT_ABLATIONS = ("mdk_only", "pard_only")
PARD_OVERRIDES = ("p1", "p2", "p3", "p4")
PARD_ABLATIONS = ("p0", *PARD_OVERRIDES, "p5")


def ablation_config(name):
    if name in COMPONENT_ABLATIONS:
        return load_config(f"{name}.json")

    # P0 is the MDK-only model. P5 uses the same settings as the full model.
    base = "mdk_only.json" if name == "p0" else "map_vmunet.json"
    config = load_config(base)
    if name in PARD_OVERRIDES:
        overrides = load_config(f"pard/{name}.json")
        config["source_preset"] = overrides["source_preset"]
        for section in ("model", "training"):
            config[section].update(overrides.get(section, {}))
    config["name"] = name
    return config


def main():
    parser = run_parser(__doc__)
    parser.add_argument(
        "--config", required=True, type=str.lower,
        choices=COMPONENT_ABLATIONS + PARD_ABLATIONS,
        help="MDK/PARD component (mdk_only, pard_only) or PARD internal configuration (P0 to P5)",
    )
    args = parser.parse_args()
    run(parser, args, ablation_config(args.config))


if __name__ == "__main__":
    main()
