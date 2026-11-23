# SPDX-License-Identifier: Apache-2.0
"""Train or evaluate the complete MAP-VMUnet model."""

from map_vmunet.runner import load_config, run, run_parser


def main():
    parser = run_parser(__doc__)
    args = parser.parse_args()
    run(parser, args, load_config("map_vmunet.json"))


if __name__ == "__main__":
    main()
