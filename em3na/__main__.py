"""
Automated DNA/RNA modeling from cryo-EM maps
Tao Li et al.
"""

def main():
    import sys
    import time
    import platform
    # Check platform
    if platform.system() != "Linux":
        print("# WARN Your system is -> {}".format(platform.system()))
        print("# WARN This program only support Linux systems (tested on CentOS 7)")
        print("# WARN This program will still run, but at any time it will crash")
        time.sleep(1)

    import argparse
    import warnings
    import em3na

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"em3na {em3na.__version__}",
    )

    # Suppress some warnings
    warnings.filterwarnings("ignore")

    import em3na.build
    import em3na.pipeline.pred
    import em3na.pipeline.eval
    import em3na.pipeline.get_qscore

    modules = {
        "build": em3na.build,
        "pred": em3na.pipeline.pred,
        "eval": em3na.pipeline.eval,
        "qscore": em3na.pipeline.get_qscore, 
    }

    subparsers = parser.add_subparsers(title="Modules",)
    subparsers.required = "True"

    for key in modules:
        module_parser = subparsers.add_parser(
            key,
            description=modules[key].__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        modules[key].add_args(module_parser)
        module_parser.set_defaults(func=modules[key].main)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
