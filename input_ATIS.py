#-*- coding:utf-8 -*-
import requests
import os

import math
import numpy as np

import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) #


def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)

locations = ['Tutorials/SLUHandsOn', 'Examples/LanguageUnderstanding/ATIS/BrainScript']

data = {
  'train': { 'file': 'atis.train.ctf', 'location': 0 },
  'test': { 'file': 'atis.test.ctf', 'location': 0 },
  'query': { 'file': 'query.wl', 'location': 1 },
  'slots': { 'file': 'slots.wl', 'location': 1 }
}

for item in data.values():
    location = locations[item['location']]
    path = os.path.join('..', location , item['file'])
    print 'location : ' , location
    print 'path : ' , path
    print 'item : ', item['file']

    if os.path.exists(path): #cntk 을 바로 다운로드 받았을때를 대비해서
        print("Reusing locally cached:", item['file'])
        # Update path
        item['file'] = path

    elif os.path.exists(os.path.join('./data/ATIS' , item['file'])):# local에서 코드를 짜서 실행할 때를 대비해서
        print("Reusing locally cached:", item['file'])
    else:
        print("Starting download:", item['file'])
        url = "https://github.com/Microsoft/CNTK/blob/release/2.2/%s/%s?raw=true"%(location, item['file'])
        print 'url :' ,url
        download(url, os.path.join('./data/ATIS',item['file']))
        print("Download completed")


vocab_size = 943 ; num_labels = 129 ; num_intents = 26
def create_reader(path, is_training):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
         query         = C.io.StreamDef(field='S0', shape=vocab_size,  is_sparse=True),
         intent_unused = C.io.StreamDef(field='S1', shape=num_intents, is_sparse=True),
         slot_labels   = C.io.StreamDef(field='S2', shape=num_labels,  is_sparse=True)
     )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

reader = create_reader(os.path.join('./data/ATIS' , data['train']['file']), is_training=True)
reader.streams.keys()