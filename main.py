#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, make_response
from flask import json
from flask import request
from flask import render_template
import os
import numpy as np
import pandas as pd
import csv
import librosa

app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def method1():
    if request.method == "POST":
        f = request.files['audio_data']
        audio_file = 'audio.wav'
        with open(audio_file, 'wb') as audio:
            f.save(audio)
            f.close()

            # Creating a header for our CSV file.
            header = 'file_name chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
            for i in range(1, 21):
                header += f' mfcc{i}'
            header += ' label'
            header = header.split()

            file = open('test.csv', 'w', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(header)
            y, sr = librosa.load(audio_file, mono=True, duration=30)
            rmse = librosa.feature.rms(y=y)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{audio_file} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f''
            file = open('test.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

            # Read the csv file back
            df = pd.read_csv('test.csv')
            print(df.head(5))
            output = df['chroma_stft'].to_list()
            response = app.response_class(
                response=json.dumps({'chroma_stft': output}),
                status=200,
                mimetype='application/json'
            )
            return response
    else:
        return render_template('index.html', output1="CSV DATA HERE")


if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.run(debug=False)
