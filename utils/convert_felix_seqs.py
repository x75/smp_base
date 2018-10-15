"""convert_felix_seqs.py

Convert felix' sequence data into ppydata smp_graphs pickled dict format and do conversions along the way
"""

import argparse, pickle
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    assert args.datafile is not None
    print('datafile', args.datafile)
    data = np.genfromtxt(args.datafile, delimiter = ' ')
    next_hundred_below = (data.shape[0]/100)*100
    data = data[:next_hundred_below]
    print("    data", data.shape)

    # save to ppydata style pickled dict
    ppydata = {}
    ppydata['x'] = np.roll(data, 25, axis = 0).reshape(data.shape + (1, ))
    ppydata['y'] = data.reshape(data.shape + (1, ))

    for k, v in list(ppydata.items()):
        print('    ppydata.%s = %s' % (k, v.shape))
    
    pickle.dump(ppydata, open(args.datafile + '.pickle', 'wb'))


    # save as wav
    from scipy.io import wavfile
    # print "data.dtype", data.dtype
    data /= np.max(np.abs(data))
    data *= 32767
    data = np.vstack((data for i in range(10)))
    wavfile.write(args.datafile + '.wav', 44100, data.astype(np.int16))
    
    plt.plot(data[:,0], data[:,1], 'k-o', alpha = 0.2)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', type = str, default = None, help = 'Datafile to load for processing')
    
    args = parser.parse_args()

    main(args)
