import model_cifar10 as model
import numpy as np
import os
from IPython import embed

exps = [
]
maxs = []

for exp in exps:
    base_dir = '/data/whyjay/NIPS2017/'
    sample_dir = base_dir + exp
    samples = sorted(list(set([int(s.split('_')[-1][:-4]) for s in os.listdir(sample_dir) if 'samples' in s])))
    print "num samples : %d" % (len(samples))
    print sample_dir

    mean_stddev = np.zeros((len(samples),2))
    for i, s in enumerate(samples):
        #with open(sample_dir + '/samples_rec_%d.npy'%s) as f:
        with open(sample_dir + '/samples_%d.npy'%s) as f:
            images = np.load(f)
        images = np.split(images, images.shape[0])
        images = [(im.reshape(im.shape[1:]) + 1)*255./2 for im in images]

        mean, stddev = model.get_inception_score(images)
        print "---------- SCORE in %s ------------" % s
        print "%f, %f"%(mean, stddev)
        mean_stddev[i,0] = mean
        mean_stddev[i,1] = stddev

    max_idx = mean_stddev[:,0].argmax()
    maxs.append((exp, mean_stddev[max_idx,0], mean_stddev[max_idx,1], samples[max_idx]))
    print 'MAX = %f, at %d' % (mean_stddev[max_idx,0], samples[max_idx])

    with open(sample_dir+'/scores.npy', 'w') as f:
        np.save(f, mean_stddev)
print maxs
