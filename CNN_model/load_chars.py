import numpy as np
import os
import fnmatch
import re
from PIL import Image
import pickle
import configparser
import keras
from dataload import convert_to_pixel_array

config = configparser.ConfigParser()
config.read('settings.ini')

# Path to datasets
dataset_dir = config['paths']['DATASET_DIRECTORY']

"""
Downloaded the NIST dataset as the by-class version.
The folder names are the hex version of the character.
Example: 4A = J
"""
nist_fname = 'nist.p'

"""
Each of the chars74k dataset classes are organized the same way.
Folders Sample001-Sample010 are digits 0-9.
Folders Sample011-Sample036 are uppercase letters A-Z
Folders Sample037-Sample0-62 are lowercase letters a-z
"""
char74k_pname = 'char74k.p'
chars_fnt_fname = 'chars_fnt.p'
chars_hnd_fname = 'chars_hnd_fname.p'
chars_img_fname = 'chars_img_fname.p'

class Dataloader:
    """
    Character labels are ord(char) to ensure consistent labeling
    between datasets and easy conversion to one-hot
    """
    def __init__(self, img_height, img_width):
        self.nist_path = dataset_dir + 'nist/by_class/'
        self.chars74k = dataset_dir + 'English/'
        self.chars_fnt_path = self.chars74k + 'Fnt/'
        self.chars_hnd_path = self.chars74k + 'Hnd/Img/'
        self.chars_img_path = self.chars74k + 'Img/'
        self.height = img_height
        self.width = img_width

    def load_all(self):
        data = self.load_nist()
        char74k_data = self.load_char74k()

        for key, value in char74k_data.items():
            data[key]['points'].extend(value['points'])

        return data

    def load_nist(self):
        nist_data = {}
        try:
            nist_data = pickle.load(open(nist_fname, 'rb'))
            print('nist loaded from {}'.format(nist_fname))
        except:
            nist_data = self.walk_nist()
            pickle.dump(nist_data, open(nist_fname, 'wb'))
            print('nist saved to ', nist_fname)
        return nist_data

    def load_char74k(self):
        char74k_data = {}
        try:
            char74k_data = pickle.load(open(char74k_pname, 'rb'))
            print('char74k loaded from {}'.format(char74k_pname))
        except:
            char74k_data = self.walk_char74k('fnt')
            hnd = self.walk_char74k('hnd')
            for key, value in hnd.items():
                char74k_data[key]['points'].extend(value['points'])
            img = self.walk_char74k('img')
            for key, value in img.items():
                char74k_data[key]['points'].extend(value['points'])
            if (len(char74k_data['a']['points']) == 0):
                print('Error: no images loaded')
                return
            pickle.dump(char74k_data, open(char74k_pname, 'wb'))
            print('char74k saved to ', char74k_pname)
        return char74k_data

    def walk_nist(self):
        data = {}
        for root, dirnames, filenames in os.walk(self.nist_path):
            sep = os.path.sep
            root_split = root.replace('\\', sep).replace('/', sep).split(sep) # Fixes inconsistency with path separator in windows
            if (root_split[-1].startswith('hsf_')):
                char = chr(int(root_split[-2], 16))
                if (char not in data):
                    data[char] = {}
                    data[char]['id'] = ord(char)
                    data[char]['points'] = []
                for filename in filenames:
                    if filename.endswith('.png'):
                        try:
                            image_path = os.path.join(root, filename)
                            point = {}
                            point['filename'] = filename
                            point['image_path'] = image_path
                            point['pixel_array'] = convert_to_pixel_array(image_path, self.width, self.height)
                            data[char]['points'].append(point)
                        except:
                            print('image not valid: {}'.format(filename))
                print('Loaded {} images for char {}'.format(len(data[char]['points']), char))
        if ('a' not in data or len(data['a']['points']) == 0):
            print('Error: no images loaded')
            exit()
        return data

    def walk_char74k(self, char_type):
        cpath = ''
        fname_reg = 'Sample(\d\d\d)'
        data = {}
        if (char_type == 'fnt'):
            cpath = self.chars_fnt_path
        elif (char_type == 'hnd'):
            cpath = self.chars_hnd_path
        else:
            cpath = self.chars_img_path
        print(cpath)
        for root, dirnames, filenames in os.walk(cpath):
            foldername = root.split(os.path.sep)[-1]
            chartype_search = re.search(fname_reg, foldername)
            if (chartype_search and len(chartype_search.groups()) == 1):
                chartype = int(chartype_search.group(1))
                char = self.char74k_num_to_char(chartype)
                if (char not in data):
                    data[char] = {}
                    data[char]['id'] = ord(char)
                    data[char]['points'] = []
                for filename in filenames:
                    if filename.endswith('.png'):
                        try:
                            image_path = os.path.join(root, filename)
                            point = {}
                            point['filename'] = filename
                            point['image_path'] = image_path
                            point['pixel_array'] = convert_to_pixel_array(image_path, self.width, self.height)
                            data[char]['points'].append(point)
                        except:
                            print('image not valid: {}'.format(filename))
                print('Loaded {} images for char {}'.format(len(data[char]['points']), char))
        return data

    def char74k_num_to_char(self, num):
        mapping = {
                1: '0',
                2: '1',
                3: '2',
                4: '3',
                5: '4',
                6: '5',
                7: '6',
                8: '7',
                9: '8',
                10: '9',
                11: 'A',
                12: 'B',
                13: 'C',
                14: 'D',
                15: 'E',
                16: 'F',
                17: 'G',
                18: 'H',
                19: 'I',
                20: 'J',
                21: 'K',
                22: 'L',
                23: 'M',
                24: 'N',
                25: 'O',
                26: 'P',
                27: 'Q',
                28: 'R',
                29: 'S',
                30: 'T',
                31: 'U',
                32: 'V',
                33: 'W',
                34: 'X',
                35: 'Y',
                36: 'Z',
                37: 'a',
                38: 'b',
                39: 'c',
                40: 'd',
                41: 'e',
                42: 'f',
                43: 'g',
                44: 'h',
                45: 'i',
                46: 'j',
                47: 'k',
                48: 'l',
                49: 'm',
                50: 'n',
        51: 'o',
                    52: 'p',
                    53: 'q',
                    54: 'r',
                    55: 's',
                    56: 't',
                    57: 'u',
                    58: 'v',
                    59: 'w',
                    60: 'x',
                    61: 'y',
                    62: 'z'
                }
        return mapping.get(num, False)
