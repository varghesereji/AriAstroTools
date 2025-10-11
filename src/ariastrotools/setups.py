import argparse


def read_args():
    '''
    Read the argument while execution.
    '''
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--fnames", nargs="+", required=True,
                        help="Input file names")
    parent.add_argument("--output", required=True,
                        help="Output file name")
    parent.add_argument("--flux", nargs="+", default=0,
                        help="Extensions of flux")
    parent.add_argument("--var", nargs="+", default=None,
                        help="Extensions of variance")
    parent.add_argument("--wl", nargs="+", default=None,
                        help="Extensions of wavelength")

    parser = argparse.ArgumentParser(description="Input data to combine")

    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       help="Choose mode")
    binary_parser = subparsers.add_parser("operation", parents=[parent],
                                          help="Binary operations on data")
    binary_parser.add_argument("operator",
                               choices=["+", "-", "*", "/"],
                               help="Binary operation (+,-,*,/)")


    # For combining
    combine_parser = subparsers.add_parser("combine", parents=[parent],
                                           help="Combine multiple data files")
    combine_parser.add_argument("method",
                                choices=["mean", "median", "biweight",
                                         "weightedavg"],
                                help="Method to combine data")

    combine_parser.add_argument(
        '--instrument',
        type=str, default=None,
        help="If the data is from any specific instrument (eg:NEID)"
    )

    return parser

# End
