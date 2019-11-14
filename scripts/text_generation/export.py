"""
Export the Text Generation Model
==============================================

This example shows how to export a pre-trained language model on wikitext-2 in Gluon NLP Toolkit model
zoo
"""

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=missing-docstring
import argparse

import numpy as np
import mxnet as mx
import gluonnlp as nlp

import model # local 'model' module with the addition of GPT-2

nlp.utils.check_version('0.7.1')

parser = argparse.ArgumentParser(description='Generate sentences by beam search. '
                                             'We load a LSTM model that is pre-trained on '
                                             'WikiText as our encoder.')
parser.add_argument('--lm-model', type=str, default='awd_lstm_lm_1150',
                help='type of the pre-trained model to load, can be "standard_lstm_lm_200", '
                '"standard_lstm_lm_650", "standard_lstm_lm_1500", '
                '"awd_lstm_lm_1150", etc.')

args = parser.parse_args()
ctx = mx.cpu()

print(args)


# Define the decoder function, we use log_softmax to map the output scores to log-likelihoods
# Also, we transform the layout to NTC
class LMDecoder:
    def __init__(self, net):
        self.net = net

    def __call__(self, inputs, states):
        outputs, states = self.net(mx.nd.expand_dims(inputs, axis=0), states)
        return outputs[0], states

    def state_info(self, *arg, **kwargs):
        return self.net.state_info(*arg, **kwargs)

class GPT2Decoder(LMDecoder):
    def __call__(self, inputs, states):
        inputs = mx.nd.expand_dims(inputs, axis=1)
        out, new_states = self.net(inputs, states)
        out = mx.nd.slice_axis(out, axis=1, begin=0, end=1).reshape((inputs.shape[0], -1))
        return out, new_states

def get_decoder_vocab(lm_model):
    if lm_model.startswith('gpt2'):
        dataset_name = 'openai_webtext'
        decoder_cls = GPT2Decoder
    else:
        dataset_name = 'wikitext-2'
        decoder_cls = LMDecoder
    lm_model, vocab = model.get_model(name=lm_model,
                                      dataset_name=dataset_name,
                                      pretrained=True,
                                      ctx=ctx)
    decoder = decoder_cls(lm_model)
    return decoder, vocab


###############################################################################
#                            Prepare dummy input data                         #
###############################################################################



def get_decoder_model(lm_model):
    if lm_model.startswith('gpt2'):
        dataset_name = 'openai_webtext'
        decoder_cls = GPT2Decoder
    else:
        dataset_name = 'wikitext-2'
        decoder_cls = LMDecoder
    lm_model, vocab = model.get_model(name=lm_model,
                               dataset_name=dataset_name,
                               pretrained=True,
                               ctx=ctx)
    return lm_model, vocab


def export():
    print ('Doing export for ', args.lm_model)
    decoder, vocab = get_decoder_vocab(args.lm_model)
    net = decoder.net
    net.hybridize()

    inputs = mx.nd.zeros((1,1024), mx.cpu())
    net(inputs)
    #net.export("carin-gpt", epoch=0)



if __name__ == '__main__':
    export()
