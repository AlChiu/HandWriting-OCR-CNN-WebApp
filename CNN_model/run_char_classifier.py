from cnn_model_chars import CharacterClassifier
import pickle

char_ids = pickle.load(open('char_ids.p', 'rb'))

classifier = CharacterClassifier(32, 32, char_ids, modelPath = 'model-62char.h5')

dataset_path = '../../../Dataset/'
nist_path = dataset_path + 'nist/by_class/'
char74k_path = dataset_path + 'English/'
char74k_fnt_path = char74k_path + 'Fnt/'
char74k_hnd_path = char74k_path + 'Hnd/Img/'
char74k_badimg_path = char74k_path + 'Img/BadImag/Bmp/'

images = [
        (nist_path + '4a/hsf_0/hsf_0_00000.png', 'J'),
        (nist_path + '30/hsf_0/hsf_0_00000.png', '0'),
        (char74k_badimg_path + 'Sample001/img001-00001.png', '0'),
        (char74k_badimg_path + 'Sample001/img001-00010.png', '0'),
        (char74k_badimg_path + 'Sample005/img005-00006.png', '4'),
        (char74k_badimg_path + 'Sample035/img035-00002.png', 'Y'),
        (char74k_badimg_path + 'Sample044/img044-00004.png', 'h'),
        (char74k_fnt_path + 'Sample062/img062-00001.png', 'z'),
        (char74k_fnt_path + 'Sample056/img056-00001.png', 't'),
        (char74k_fnt_path + 'Sample047/img047-00001.png', 'k'),
        (char74k_hnd_path + 'Sample019/img019-001.png', 'I'),
        (char74k_hnd_path + 'Sample036/img036-001.png', 'Z'),
        ]

for image in images:
    print('Actual: ', image[1])
    print(classifier.classify_image(image[0]))
