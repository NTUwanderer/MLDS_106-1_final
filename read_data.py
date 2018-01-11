import os

synth_dirs = ['DeepQ-Synth-Hand-01/data', 'DeepQ-Synth-Hand-02/data']
synth_cDirs = []
for synth_dir in synth_dirs:
    synth_cDirs += [os.path.join(synth_dir, data_dir)  for data_dir in sorted(os.listdir(synth_dir))]

real_cDirs = ['DeepQ-Vivepaper/data/air', 'DeepQ-Vivepaper/data/book']

synth_image_label_pair = []
real_image_label_pair = []

def loadPairs(cDir):
    imgDir = os.path.join(cDir, 'img')
    labelDir = os.path.join(cDir, 'label')

    return zip([os.path.join(imgDir, img) for img in sorted(os.listdir(imgDir))], [os.path.join(labelDir, label)  for label in sorted(os.listdir(labelDir))])

for cDir in synth_cDirs:
    synth_image_label_pair += loadPairs(cDir)

for cDir in real_cDirs:
    real_image_label_pair += loadPairs(cDir)

print ('len: ', len(synth_image_label_pair), len(real_image_label_pair))
# print (':100 : ', synth_image_label_pair[:100])
# print (':100 : ', real_image_label_pair[:100])

def getSynthPairs():
    return synth_image_label_pair

def getRealPairs():
    return real_image_label_pair


