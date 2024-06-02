from flask import Flask, request, render_template, redirect, url_for, jsonify
from moviepy.editor import VideoFileClip, concatenate_videoclips, concatenate_audioclips
from sklearn.cluster import KMeans
import librosa
import numpy as np
import pandas as pd
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
app = Flask(__name__, static_folder='./static')
@app.route('/')
def highlight():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part' 
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    file.save('C:\\Users\\pavitra\\Desktop\\AH_generator\\static\\Audios\\' + file.filename)
    video_path = 'C:\\Users\\pavitra\\Desktop\\AH_generator\\static\\Audios\\' + file.filename
    print("Video is Uploaded Successfully")
    return generate_highlights(video_path)
def generate_highlights(video_path):
    def save_audio(video_path):
        video = VideoFileClip(video_path)
        audio = video.audio
        print("Video To Audio Convertion Successfully")
        save_path = "static/Audios/sample.wav"
        audio.write_audiofile(save_path)
        return save_path
    def threshold(filename):
        def extract_features(audio_chunk, sr):
            mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=13)
            energy = np.sum(np.abs(audio_chunk)**2)
            centroid = librosa.feature.spectral_centroid(y=audio_chunk, sr=sr)[0, 0]
            spread = librosa.feature.spectral_bandwidth(y=audio_chunk, sr=sr)[0, 0]
            features = np.concatenate((mfcc.mean(axis=1), [energy, centroid, spread]))
            return features
        y, sr = librosa.load(filename, sr=None)
        chunk_duration = 5
        chunk_size = chunk_duration * sr
        X = []
        num_chunks = len(y) // chunk_size
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            audio_chunk = y[start_idx:end_idx]
            features = extract_features(audio_chunk, sr)
            X.append(features)
        X = np.array(X)
        num_clusters = 2
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_
        cluster_means = [np.mean(X[labels == i], axis=0) for i in range(num_clusters)]
        highlight_cluster = np.argmax([cluster_mean[-3] for cluster_mean in cluster_means])
        threshold_value = cluster_means[highlight_cluster][-3]
        return threshold_value
    audio_path = save_audio(video_path)
    x, sr = librosa.load(audio_path)
    max_slice = 5
    window_length = max_slice * sr
    a = x[2*window_length:3*window_length]
    energy = sum(abs(a**2))
    df = pd.DataFrame(columns=['energy', 'start', 'end'])
    thresh = threshold(audio_path)
    row_index = 0
    for i in range(len(a)):
        value = energy
        if value >= thresh:
            i = np.where(energy == value)[0]
            df.loc[row_index, 'energy'] = value
            df.loc[row_index, 'start'] = i[0] * 5
            df.loc[row_index, 'end'] = (i[0]+1) * 5
            row_index += 1
    temp = []
    i = 0
    j = 0
    n = len(df) - 2
    m = len(df) - 1
    while i <= n:
        j = i+1
        while j <= m:
            if df['end'][i] == df['start'][j]:
                df.loc[i, 'end'] = df.loc[j, 'end']
                temp.append(j)
                j = j+1
            else:
                i = j
                break
    df.drop(temp, axis=0, inplace=True)
    output_directory = "static/videos"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    start = np.array(df['start'])
    end = np.array(df['end'])
    for i in range(len(df)):
        if i != 0:
            start_lim = start[i] - 5
        else:
            start_lim = start[i]
        end_lim = end[i]
 
        filename = "highlight" + str(i+1) + ".mp4"
        output_path = os.path.join(output_directory, filename)
        ffmpeg_extract_subclip(video_path, start_lim, end_lim, targetname=output_path)
    highlight_paths = [os.path.join(output_directory, filename) for filename in os.listdir(output_directory) if filename.startswith("highlight")]
    highlight_clips = [VideoFileClip(path) for path in highlight_paths]
    final_clip = highlight_clips[0]
    for i in range(1, len(highlight_clips)):
        transition_duration = 1
        final_clip = final_clip.fadeout(transition_duration)
        final_clip = concatenate_videoclips([final_clip, highlight_clips[i]])
    final_clip = final_clip.set_audio(concatenate_audioclips([clip.audio for clip in highlight_clips]))
    final_folder = os.path.join(output_directory, "final")
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)
    final_video_path = os.path.join(final_folder, "final_highlights.mp4")
    final_clip.write_videofile(final_video_path, codec="libx264", fps=30)
    return redirect(url_for('show_video'))
@app.route('/show_video', methods=['GET'])
def show_video():
    return render_template('video.html')
@app.route('/check_video_generation')
def check_video_generation():
    final_video_path = url_for('static', filename='/videos/final/final_highlights.mp4')
    if final_video_path:
        return jsonify({"video_path": final_video_path})
    else:
        return jsonify({"status": "not_ready"})
if __name__ == '__main__':
    app.run(debug=True)
