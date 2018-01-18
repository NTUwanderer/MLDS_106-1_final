import os

def loadPairs(cDir):
    imgDir = os.path.join(cDir, 'img')
    labelDir = os.path.join(cDir, 'label')

    return zip([os.path.join(imgDir, img) for img in sorted(os.listdir(imgDir))], [os.path.join(labelDir, label)  for label in sorted(os.listdir(labelDir))])

def loadEmptyPairs(cDir):
    imgDir = cDir

    return zip([os.path.join(imgDir, img) for img in sorted(os.listdir(imgDir))], [0  for img in sorted(os.listdir(imgDir))])


def getSynthPairs(img_dir):
    synth_dirs = [img_dir + 'DeepQ-Synth-Hand-01/data', img_dir + 'DeepQ-Synth-Hand-02/data']
    synth_cDirs = []
    for synth_dir in synth_dirs:
        synth_cDirs += [os.path.join(synth_dir, data_dir)  for data_dir in sorted(os.listdir(synth_dir))]

    synth_image_label_pair = []

    for cDir in synth_cDirs:
        synth_image_label_pair += loadPairs(cDir)

    return synth_image_label_pair

def getRealPairs(img_dir):
    real_cDirs = [img_dir + 'DeepQ-Vivepaper/data/air', img_dir + 'DeepQ-Vivepaper/data/book']

    real_image_label_pair = []

    for cDir in real_cDirs:
        real_image_label_pair += loadPairs(cDir)

    return real_image_label_pair

def getRealTestPairs():
    #real_cDirs = ['data/DeepQ-Vivepaper/frame/air', 'data/DeepQ-Vivepaper/frame/book']
    real_cDirs = ['test_real/']

    real_image_label_pair = []

    for cDir in real_cDirs:
        real_image_label_pair += loadEmptyPairs(cDir)

    return real_image_label_pair


