#!/usr/bin/env bash

feather_name="../data/{psrname}_12p5y.feather"


python create_feather_from_par_tim.py --feather-name=$feather_name \
                                      --parfile-directory=$parfile_directory \
                                      --timfile-directory=$timfile_directory

