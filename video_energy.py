from array import array
import os
import  numpy as np
import imageio
from  Stats import graphs
imageio.plugins.ffmpeg.download()
import  sys
from moviepy.editor import *
import pygame

import uuid

import random
import  re
import librosa as lib
import pydub as dub
from pydub import  AudioSegment
import math as m
import cv2 as cv2
import cv2wrap
import numpy as np

if __name__ == '__main__':

    stats = graphs()
    stats.video_plot(sys.argv[1])
