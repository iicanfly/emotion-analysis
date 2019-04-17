#-*- encoding=utf-8 -*-
"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from FlaskWebProject1 import app
from flask import render_template, redirect, request, flash, get_flashed_messages
from flask_login import login_user, logout_user, login_required, current_user
import hashlib
import random
import json
import numpy as np
import random
import sys
import io
import json
sys.path.append('../2_types/')

import NeuralNetwork
from filehandle_train import *
import pandas 

@app.route('/')
@app.route('/home',methods={'get', 'post'})
def home():
    """Renders the home page."""
    comment = request.values.get('comment')
    comment_print = request.values.get('comment')
    comment_result = 'None'
    comment_type = 'None'
    if comment:
        comment = comment.split(' ')
        wordSetLen = 23151
        labelNum = 1
        net = NeuralNetwork.Network([wordSetLen, 64, 64, 64, labelNum])
        j = 9
        net = NeuralNetwork.load("../2_types/parameter/epoch%s_sizes.txt" % j,
                                 "../2_types/parameter/epoch%s_weights.txt" % j,
                                 "../2_types/parameter/epoch%s_biases.txt" % j,
                                 "../2_types/parameter/epoch%s_cost.txt" % j)
        wordLib = buildWordLib("../2_types/2/train_data_last.txt")
        libTmp = wordLib.copy()
        for word in comment:
            if word in libTmp:
                libTmp[word] = 1
        a = np.mat(list(libTmp.values())).transpose()
        comment_result = net.feedforward(a)
        if comment_result > 0.5:    
            comment_type = 'good'
        else:
            comment_type = 'bad'
    return render_template(
        'home.html',
        comment_print = comment_print,
        comment_result = comment_result,
        comment_type = comment_type
    )


    
