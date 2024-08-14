#!/usr/bin/env python
# -*- coding: utf-8 -*-

from settings import load_pulsars
import json

datadir = '../../../../nanograv_data/data-12.5yr/NANOGrav_12yv4/narrowband'
noisedict = '../../../../nanograv_data/12p5_data/channelized_12p5yr_v3_full_noisedict.json'
noisedict = json.load(open(noisedict, 'r'))

load_pulsars(datadir, outdir='../data/', PINT=True, noisedict=noisedict)
