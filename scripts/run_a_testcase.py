from hmm import hmm
import sys
import logging

if __name__ == '__main__':
    logfile = 'logger.txt'
    try:
        logfile = sys.argv[2]
    except:
        pass
    logging.basicConfig(filename=logfile, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    sim = hmm.simulation(sys.argv[1])
    if sim.only_md:
        sim.run_md()
    else:
        sim.run_hmm()
