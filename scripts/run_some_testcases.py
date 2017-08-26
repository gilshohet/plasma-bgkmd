from hmm import hmm
import sys
import os
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    names = ['/home/shohet/bgk-md/input/md_temperature_1000_2',
             '/home/shohet/bgk-md/input/md_temperature_900_2',
             '/home/shohet/bgk-md/input/md_temperature_850_2',
             '/home/shohet/bgk-md/input/md_temperature_800_2',
             '/home/shohet/bgk-md/input/md_temperature_775_2']

    for name in names:
        os.chdir('/home/shohet/bgk-md/scripts')
        sim = hmm.simulation(name)
        if sim.only_md:
            sim.run_md()
        else:
            sim.run_hmm()
