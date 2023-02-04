import sys
import os
import io
import json
import argparse
import numpy as np
import pandas as pd
import h5py
from pprint import pprint
from collections import OrderedDict

from pygama import flow
from pygama.raw.build_raw import build_raw
from pygama.dsp.build_dsp import build_dsp


def main():
    doc = ""
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'

    arg('-r', '--runs', nargs=1, type=int,
            help="list of files to process from runDB.json (-r #) ")
    arg('-c', '--config', nargs='*', type=str, help='configuration json file (-c config.json).')

    arg('--d2r', action=st, help='run daq_to_raw')
    arg('--r2d', action=st, help='run raw_to_dsp')


    args = par.parse_args()

    query = args.runs[0]
    config_file = args.config[0]


    cwd = os.getcwd()
    file = cwd + '/coherent.json'
    with open(file, 'r') as read_file:
        data = json.load(read_file)

    config = cwd + '/' + config_file
    with open(config, 'r') as read_file:
        configure = json.load(read_file)

    runsFile = cwd + '/runDB.json'
    with open(runsFile, 'r') as read_file:
        run_lists = json.load(read_file)

    run_list = run_lists[str(query)]['run_list']
    if isinstance(run_list, str):
        idx = run_list.find('-')
        run_list = [x for x in range(int(run_list[0:idx]), int(run_list[idx+1:])+1)]



    #run_list = [x for x in range(1202,1231)]
    if args.d2r:
        for run in run_list:
            #dataFile = data['daq_dir'] + '/Run' + str(run) + '.gz'

            dataFile = data['daq_dir'] + '/Run' + str(run) + ".gz"
            outFile = data['raw_dir'] + '/Run' + str(run) + '.lh5'
            configure["ORSIS3316WaveformDecoder"]["Det61"]["out_stream"] = outFile
            configure["ORSIS3316WaveformDecoder"]["veto61"]["out_stream"] = outFile

            #try:
            build_raw(dataFile, data['stream_type'], configure, overwrite=True)
            #except:
                #print("Run does not exist")
                #build_raw(dataFile, overwrite=True)
    if args.r2d:
        for run in run_list:
            print(run)
            chan_file = cwd + "/chan_config.json"
            conf_file = cwd + "/dspConfigFiles/dsp_config.json"
            with open(conf_file) as f:
                dsp = json.load(f, object_pairs_hook=OrderedDict)
            with open(chan_file) as f:
                chan_conf = json.load(f, object_pairs_hook=OrderedDict)
        
            

            dataFile = data['raw_dir'] + '/Run' + str(run) + '.lh5'
            outFile = data['dsp_dir'] + '/Run' + str(run) + '.lh5'
            build_dsp(dataFile, outFile, chan_config = chan_conf, write_mode = 'r')

            #build_dsp(dataFile, outFile, dsp_config = dsp)


if __name__ == "__main__":
    main()