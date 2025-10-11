#!/usr/bin/env python3

import logging


from .logger import logger

from .setups import read_args


from .handle_frame import operate_process
from .handle_frame import combine_process
from .spectral_utils import combine_spectra


def setup_logging():
    logging.basicConfig(
        filename='ariastro_comb.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )


def process_inputs(file2: str):
    # Check if file2 is a numbr or string.
    try:
        val = float(file2)
        is_number = True
    except ValueError:
        is_number = False
    if is_number:
        return val
    else:
        return file2


def main():
    parser = read_args()
    print(parser)
    args = parser.parse_args()
    fnames = args.fnames
    logger.info("Starting the pipeline")
    logger.info("Flux extensions: {}".format(args.flux))
    if args.var is not None:
        logger.info("Variance extensions: {}".format(args.var))
    if args.wl is not None:
        logger.info("Wavelength extensions: {}".format(args.wl))

    if args.mode == 'combine':
        print(fnames)
        if args.wl is None:
            combine_process(fnames,
                            args.output,
                            method=args.method,
                            fluxext=args.flux,
                            varext=args.var,
                            instrument=args.instrument
                            )
        else:
            combine_spectra(fnames,
                            opfilename=args.output,
                            method=args.method,
                            fluxext=args.flux,
                            varext=args.var,
                            wlext=args.wl)
            
    elif args.mode == 'operation':
        file1, file2 = fnames

        file2 = process_inputs(file2)
        operate_process(file1, file2,
                        args.output,
                        args.operator,
                        args.flux,
                        args.var)


if __name__ == '__main__':
    setup_logging()

# End
