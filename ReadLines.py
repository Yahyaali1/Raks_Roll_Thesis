from array import array
import os
import  numpy as np
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import *
import pygame
import sys
import uuid
import nltk
from nltk.corpus import PlaintextCorpusReader
import random
import librosa as lib
from matplotlib import pyplot as plt
import pydub as dub
from pydub import  AudioSegment
import math as m

corpus_root = 'C:/Users/IBM/Desktop/FYP Codes/untitled2'
wordlists = PlaintextCorpusReader(corpus_root, '.*')

words = wordlists.words('feedback.txt')
print(words)
freq_dance = nltk.FreqDist(words)
cfreq_dance_2gram = nltk.ConditionalFreqDist(nltk.bigrams(words))
cprob_dance_2gram = nltk.ConditionalProbDist(cfreq_dance_2gram, nltk.MLEProbDist)
import numpy as np


#np.set_printoptions(threshold='nan')


mazy_f=[];
moves_90={};
moves_60={};
moves_120={};
repeat_t=3
selection_t=9
beats_f=[];
bpm_range =[];
body_range=[];
count_range=[];
file_names=[];
end_string='.mov'
formate_song='mp3'
center_string='_'
file_duration=[];
combination_index=[];
song_file_in='95.mp3'
song_file_out='output.mp3'
time_crop=60000;
beat_t_labels={}

def round_beats(beats_time):
    x=0;
    for i in beats_time:
        beats_time[x]=round(beats_time[x],2)
        #beats_t.append(round(beats_time[x],2))
        x=x+1

#give time in ms
#https://github.com/jiaaro/pydub/blob/master/API.markdown
def crop_audio(song_filename,time_crop,formate_song):
    sound1 = AudioSegment.from_file(song_filename, format=formate_song)

    # first time_crop seconds in mili seconds of sound1
    sound1 = sound1[:time_crop]

    file_handle = sound1.export(song_file_out, format="mp3")
#http://librosa.github.io/librosa/generated/librosa.beat.tempo.html
def beats_array(song_filename):

    y,source_file = lib.load(song_filename)
   # onset_env = lib.onset.onset_strength(y, sr=source_file)
    #tempo = lib.beat.tempo(onset_envelope=onset_env, sr=source_file)
    #print (tempo)
    tempo,beats = lib.beat.beat_track(y=y,sr=source_file)
    beats_time=lib.frames_to_time(beats,sr=source_file)
    round_beats(beats_time)
    return beats_time,tempo
    #print (beats_t)
    # print (tempo)

    #print (beats_time[1])

# reads the data from the file to populate the respective lists
def motion_data(filename):
    file_path= filename;
    file= open(file_path);
    for line in file:
        line= line.strip();
        list= line.split(" ");
        count_range.append(list.pop());
        body_range.append(list.pop());
        bpm_range.append(list.pop());
    return
#populate the list of the files with the names based on the bpm
#populates file_names on given bpm
def list_names(bpm):
    found=False
    temp =[];
    x=0
    for i in bpm_range:
        if int(i)==bpm:
            temp.append(x);
        x=x+1;
    for i in temp:
        y=int(count_range[i]);
        for z in range(1,y+1,1):
            file_names.append(bpm_range[i]+center_string+body_range[i]+center_string+'Y'+center_string+str(z)+end_string);
            file_names.append(bpm_range[i] + center_string + body_range[i] + center_string + 'N' + center_string + str(z) + end_string);
    #print(file_names);
#calculates the duration in secs for each clip
#https://zulko.github.io/moviepy/
def file_duration_cal():
    for i in file_names:
        seq = VideoFileClip(i);
        file_duration.append(seq.duration);

#https://zulko.github.io/moviepy/
def play_videofile(file_name):
    clip = VideoFileClip(file_name)
#sum of combined indces.
def sum_seq():
    s=0;
    for i in combination_index:
        s=s+file_duration[i]
    return s
def search_dif(temp_sec,beats_f):
    x=0;
    while(x <len(beats_f) and temp_sec>=beats_f[x][0] ):
        if temp_sec==beats_f[x][0]:
            break
        x=x+1
    if x>=len(beats_f):
        x=x-1;
    # print("search next", beats_f[x][0]-temp_sec)
    return abs(beats_f[x][0]-temp_sec);

def search_next(beats_f,time):
    for i in beats_f:
        if time<= i[0]:
            # print("search next",i[1])
            return i[1]
def populate_names(): #name of the video is loaded
    global moves_60,moves_90,moves_120;
    m_60=10
    m_90=22
    m_120=28
    str_60="60_"
    str_90="90_"
    str_120="120_"
    for i in range(1,m_60+1,1):
        name=str_60+str(i)+".mov"
        seq = VideoFileClip(name);
        moves_60[name]=seq.duration

    for i in range(1,m_90+1,1):
        name=str_90+str(i)+".mov"
        seq = VideoFileClip(name);
        moves_90[name]=seq.duration

    for i in range(1,m_120+1,1):
        name=str_120+str(i)+".mov"
        seq = VideoFileClip(name);
        moves_120[name]=seq.duration
def regions(dtempo,beats_time_c):
    change=[]
    beats=[]
    change.append(beats_time_c[0])
    beats.append(dtempo[0])
    for i in range(0,len(dtempo)-1,1):
        if dtempo[i]!=dtempo[i+1]:
            change.append(beats_time_c[i])
            beats.append(dtempo[i+1])
            change.append(beats_time_c[i])
    change.append(beats_time_c[len(beats_time_c)-1])
    print (change)
    print(beats)
    return change,beats
#define regions with change[i]-change[i+1] with there valyue on corresponsing at beats[i]


def return_label(change,beat_value,local_beat):
    count =0
    for i in range(0,len(change)-1,2):
        if local_beat>= change[i] and local_beat<change[i+1]:
            return beat_value[count]
        count=count+1

def place_label(beats_loc_act,dtempo,beats_time_c):
    change=[]
    beat_value=[]
    change,beat_value =regions(dtempo,beats_time_c)

    global beat_t_labels
    beat_t_labels=np.zeros(shape=(len(beats_loc_act), 2))
    for i in range(0,len(beats_loc_act),1):
        beat_t_labels[i][0] = beats_loc_act[i]
        beat_t_labels[i][1]=return_label(change,beat_value,beats_loc_act[i])
    print(beat_t_labels)
    return beat_t_labels

#loc

#feedback functin
def getFeedBack(combination_seq):
    feedback = ""
    trueVal = 1
    print("Please watch your video and provide feedback in range for 1-5 for smoothness of the dance.\n")
    print("Enter feedback 1-5:")
    while (trueVal == 1):
        if (sys.version_info > (3, 0)):
            # Python 3 code in this block
            feedback = input("")


        feedbackInt = int(feedback)
        if ((feedbackInt > 0) & (feedbackInt < 6)):
            f = open("feedback.txt", "a+")
            for line in range(feedbackInt):
                f.write("* ")
                for i in combination_seq:
                    f.write(i.replace(".mov","") + " ")
                f.write("*\n")
            f.close()
            break
        else:
            print("Please enter a valid feedback again between 1-5 inclusive.")

def dynamic_temp(song_file_in):
    y, source_file = lib.load(song_file_in)
    tempo, beats = lib.beat.beat_track(y=y, sr=source_file)

    plot_beats=lib.frames_to_time(beats, sr=source_file)
    #convert to to 2 decimal place here
    round_beats(plot_beats)

    onset_env = lib.onset.onset_strength(y, sr=source_file)
    dtempo = lib.beat.tempo(onset_envelope=onset_env, sr=source_file, aggregate = None)
    #dynamic bpm being calculated

    for i in range(0,len(dtempo),1):
        if dtempo[i] >= 60 and dtempo[i] < (60 + 90) / 2:
            dtempo[i] = 60;
        elif dtempo[i] >= (60 + 90) / 2 and dtempo[i] < (90 + 120) / 2:
            dtempo[i] = 90;
        else:
            dtempo[i] = 120;
    #smothing of the dynamic bpm


    tempo, beats = lib.beat.beat_track(y=y, sr=source_file)
    beats_time = lib.frames_to_time(np.arange(len(dtempo)))
    #time frame of the beats that being tracked
    return place_label(plot_beats,dtempo,beats_time)


    # print(len(beats_time))
    #
    # print(beats_time)
    # print(len(dtempo))
    # plt.plot(beats_time, dtempo,linewidth=1.5)
    # plt.plot(plot_beats,np.full(len(plot_beats),100),'ro')
    # plt.show()

def return_max(array_in, last_move):
    max_prob = 0
    print(max_prob)
    index = 0
    for i in range (0,len(array_in),1):
        if combination_index.count(array_in[i][0])<repeat_t:
            max_prob = cprob_dance_2gram[last_move].prob(array_in[i][0].replace(".mov",""))
            index=i
            break

    # print("PROB samples", cprob_brown_2gram["90"].samples())

    for i in range(0, selection_t,1):
        print("range")
        print(cprob_dance_2gram[last_move.replace(".mov","")].prob(array_in[i][0].replace(".mov","")))
        if max_prob < cprob_dance_2gram[last_move.replace(".mov","")].prob(array_in[i][0].replace(".mov","")) and combination_index.count(array_in[i][0])<repeat_t:
            index = i
            print("max 1",max_prob)
            max_prob = cprob_dance_2gram[last_move.replace(".mov","")].prob(array_in[i][0].replace(".mov",""))
            print("max2",max_prob,"after")

    return array_in[index]


def mazy_dance_2(duration_song,beats_f):
    mazy_factor=[];
    sum_far=0
    a=beats_f[0][1] #type of the dance bpm we need
    if a==60:
        x=list(moves_60.keys())[random.randint(1,len(moves_60)-1)]
        y=moves_60[x]
    elif a==90:
        x = list(moves_90.keys())[random.randint(1, len(moves_90) - 1)]
        y = moves_90[x]
    else:
        x = list(moves_120.keys())[random.randint(1, len(moves_120) - 1)]
        y = moves_120[x]

    combination_index.append(x)
    sum_far=sum_far+y;

    while (sum_far<=duration_song):
        min=1000000;
        array_60=np.zeros(shape=(len(moves_60),3),dtype=object)
        array_90=np.zeros(shape=(len(moves_90), 3),dtype=object)
        array_120=np.zeros(shape=(len(moves_120), 3),dtype=object)
        index=0;
        x=0;
        c=0;
        #select the next beat based on the sequence you have combined
        next_beat=search_next(beats_f,sum_far)
        print("Next Beat",next_beat,"sum far",sum_far)
        if next_beat==60:
            for k,v in moves_60.items():  # we need to see the file of the type the beat is
                temp = sum_far + v;
                y = search_dif(temp, beats_f)
                array_60[c][0]=k
                array_60[c][1]=v
                array_60[c][2]=y
                c = c + 1
            array_60=sorted(array_60, key=lambda x: x[2])

            #pass it to lambda function
            temp2=return_max(array_60,combination_index[len(combination_index)-1])
            min=temp2[2]
            index=temp2[0]
            x=temp2[1]
        elif next_beat==90:
            for k,v in moves_90.items():  # we need to see the file of the type the beat is
                temp = sum_far + v;
                y = search_dif(temp, beats_f)
                array_90[c][0]=k
                array_90[c][1]=v
                array_90[c][2]=y
                c=c+1
            array_90=sorted(array_90, key=lambda x: x[2])

            #pass it to lambda function
            temp2=return_max(array_90,combination_index[len(combination_index)-1])
            min = temp2[2]
            index = temp2[0]
            x = temp2[1]
        else:
            for k,v in moves_120.items():  # we need to see the file of the type the beat is
                temp = sum_far + v;
                y = search_dif(temp, beats_f)
                array_120[c][0]=k
                array_120[c][1]=v
                array_120[c][2]=y
                c = c + 1
            array_120=sorted(array_120, key=lambda x: x[2])

            #pass it to lambda function
            temp2=return_max(array_120,combination_index[len(combination_index)-1])
            min = temp2[2]
            index = temp2[0]
            x = temp2[1]
        print("adding move")
        print(temp2,index,min)
        mazy_factor.append(min);
        combination_index.append(index)
        sum_far = sum_far + x;
    return mazy_factor

def mazy_dance(duration_song,beats_f):
    mazy_factor=[];
    print(len(file_duration))
    a=random.randint(1,len(file_duration)-1)
    combination_index.append(a)
    sum_far=sum_seq();
    while (sum_far<=duration_song):
        min=1000000;
        index=0;
        x=0;
        for i in file_duration: #we need to see the file of the type the beat is
            temp=sum_far+i;
            y=search_dif(temp,beats_f)
            if (y<min):
                min=y;
                index=x;
            x=x+1;
        mazy_factor.append(min);
        combination_index.append(index)
        sum_far = sum_seq();
    return mazy_factor

# list_names(90);
#crop_audio('abc.mp3',time_crop,formate_song);



def load_data():
    motion_data("Motion.txt")

#function needs to deal with the problem of dynamic temp and the numbe of beats it has, currently not adding up together.

# def smoithing_differenc(video_file1,video_file2):
#     seq = VideoFileClip(video_file1)
#     img1=seq.duration-0.1; #since it might miss out last frame therefore we need ot have this.
#     seq.save_frame("frame1.jpeg",img1)
#     seq.save_frame("frame2.jpeg", 0.0)
#     x=cv2.cvtColor(cv2.imread("frame1.jpeg"), cv2.COLOR_BGR2GRAY)
#     y=cv2.cvtColor(cv2.imread("frame2.jpeg"), cv2.COLOR_BGR2GRAY)
#
#     ret, x = cv2.threshold(x, 126, 256, cv2.THRESH_BINARY_INV)
#     cv2.imshow('',x)
#     ret, y = cv2.threshold(y, 126, 256, cv2.THRESH_BINARY_INV)
#     cv2.imshow('messigray.png',cv2.absdiff(x,y))
#     cv2.waitKey()


def main(song_file_in):
    populate_names()

    # load_data()
    global beats_f
    beats_f, temp_f = beats_array(song_file_in)
    type_temp = 0;

    sound = AudioSegment.from_file(song_file_in, format=formate_song)
    x = len(sound)
    a = x
    x = x - beats_f[0] * 1000
    sound = sound[-x:]
    print(x)
    x = x - (a - beats_f[len(beats_f) - 1] * 1000)
    print(x)
    sound = sound[:x]
    sound.export(song_file_in, format=formate_song)
    round(temp_f, 2);
    print("Bpm for this song is", temp_f)
    print("Beats are found at following locations", beats_f)
    # decide the type of file we want to load.
    if temp_f >= 60 and temp_f < (60 + 90) / 2:
        type_temp = 60;
    elif temp_f >= (60 + 90) / 2 and temp_f < (90 + 120) / 2:
        type_temp = 90;
    else:
        type_temp = 120;
    print(type_temp)
    # load the file names
    list_names(type_temp);
    file_duration_cal();

    song_input = AudioSegment.from_file(song_file_in, format=formate_song)
    duration_song = round(len(song_input) / 1000, 2)
    print("Song has following duration", duration_song)
    global mazy_f
    mazy_f = mazy_dance(duration_song, beats_f);
    print(mazy_f)
    print(combination_index)

    combination_seq = combination_index

    combination_seq_video = []
    for i in combination_seq:
        temp = VideoFileClip(i)
        combination_seq_video.append(temp);

    name = str(uuid.uuid4()) + ".mp4"
    final_clip = combination_seq_video[0];
    combination_seq_video.remove(combination_seq_video[0]);
    for i in combination_seq_video:
        final_clip = concatenate_videoclips([final_clip, i])
    audioclip = AudioFileClip(song_file_in)
    final_clip.write_videofile(name, audio=song_file_in);
    # final_clip.set_audio(audioclip)
    print(name + "Is your dance");
    getFeedBack(combination_index)
    return name;


def main2(song_file_in):
    populate_names()

    # load_data()
    global beats_f
    beats_f = dynamic_temp(song_file_in)

    sound = AudioSegment.from_file(song_file_in, format=formate_song)
    x = len(sound)
    a = x
    x = x - beats_f[0][0] * 1000
    sound = sound[-x:]
    if len(sound)/1000 >120:
        sound=sound[:120000]
    print(x)
    x = x - (a - beats_f[len(beats_f) - 1][0] * 1000)
    print(x)
    sound = sound[:x]
    sound.export(song_file_in, format=formate_song)

    # decide the type of file we want to load.
    song_input = AudioSegment.from_file(song_file_in, format=formate_song)
    duration_song = round(len(song_input) / 1000, 2)
    print("Song has following duration", duration_song)
    global mazy_f
    mazy_f = mazy_dance_2(duration_song, beats_f);
    print(mazy_f)
    print(combination_index)

    combination_seq = combination_index

    combination_seq_video = []
    for i in combination_seq:
        temp = VideoFileClip(i)
        combination_seq_video.append(temp);

    name = str(uuid.uuid4()) + ".mp4"
    final_clip = combination_seq_video[0];
    combination_seq_video.remove(combination_seq_video[0]);
    for i in combination_seq_video:
        final_clip = concatenate_videoclips([final_clip, i])
    audioclip = AudioFileClip(song_file_in)
    final_clip.write_videofile(name, audio=song_file_in);
    # final_clip.set_audio(audioclip)
    print(name + "Is your dance");

    play_videofile(name)
    getFeedBack(combination_seq)
    return name;
if __name__ == '__main__':
    audio = "audio.mp3"

    main2(audio)
    # dynamic_temp(audio)
    #smoithing_differenc("120_HAND_Y_3.mov","120_HAND_Y_3.mov")
    #class for plotting graphs
    # stats = graphs()

    # name=main(audio)
    # name="9341878f-4507-4560-a1c6-b176a39c7b03.mp4"
    #x=0.5
    #stats.plot_mazy(mazy_factor=mazy_f,width_bar=x)
    # stats.rms_plot("Bom.mp3")
    # stats.plot_show()
    # os.system('python video_energy.py '+name)
    # clip = VideoFileClip(name)
    # clip.preview();

    #stats.video_plot(name)





