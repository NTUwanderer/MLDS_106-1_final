import os

def loadPairs(cDir):
    imgDir = os.path.join(cDir, 'img')
    labelDir = os.path.join(cDir, 'label')

    return zip([os.path.join(imgDir, img) for img in sorted(os.listdir(imgDir))], [os.path.join(labelDir, label)  for label in sorted(os.listdir(labelDir))])

def getSynthPairs():
    synth_dirs = ['data/DeepQ-Synth-Hand-01/data', 'data/DeepQ-Synth-Hand-02/data']
    synth_cDirs = []
    for synth_dir in synth_dirs:
        synth_cDirs += [os.path.join(synth_dir, data_dir)  for data_dir in sorted(os.listdir(synth_dir))]

    synth_image_label_pair = []

    for cDir in synth_cDirs:
        synth_image_label_pair += loadPairs(cDir)

    return synth_image_label_pair

def getRealPairs():
    real_cDirs = ['data/DeepQ-Vivepaper/data/air', 'data/DeepQ-Vivepaper/data/book']

    real_image_label_pair = []

    for cDir in real_cDirs:
        real_image_label_pair += loadPairs(cDir)

    return real_image_label_pair


