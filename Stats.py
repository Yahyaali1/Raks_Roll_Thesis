import matplotlib.pyplot as mat
import numpy as np
import librosa as lib
import cv2 as cv2
import cv2wrap
import math as math
import statistics as stat
class graphs:
    def plot_beats(self,beat_f):
        x=np.arange(len(beat_f))

        mat.plot(x, beat_f, 'ro')
        mat.show()

    def rms_plot(self,song_file_in):
        y, sr = lib.load(song_file_in)
        x=lib.feature.rmse(y=y)
        temp=[]
        # mat.hist(x,histtype='bar',align='mid',orientation='vertical')
        # mat.show()
        # max_diff = max(x)
        # min_diff = min(x)
        # np.around(x,0)
        # #print(type(x))
        for a in x:
             #a= ((a-min_diff)/(max_diff-min_diff))
             temp.append(a)
        temp=np.asarray(temp)
        temp_2 = temp.reshape(len(temp.T),1)

        r=np.arange(0,len(temp.T),1)

        print(r)
        mat.figure(figsize=(100,100))
        mat.subplot(2,1,2)
        mat.xlabel('beats')
        mat.ylabel('energy')
        mat.plot(r, temp.T)



    def video_plot(self,video_file_in):

        cap = cv2.VideoCapture(video_file_in)
        ret, frame1 = cap.read()
        Motion_Sum=[];

        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        while(1):
            #read a next frame from the objects
            ret, frame2 = cap.read()
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #calculate flow by provding two frames from the video to the function


            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            #print(mag)

            print(hsv[...,2])
            Motion_Sum.append(np.sum(hsv[..., 2]))
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

            print(Motion_Sum)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow('frame2',bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png',frame2)
                cv2.imwrite('opticalhsv.png',bgr)
            prvs = next
        cap.release()
        cv2.destroyAllWindows()
    #
    # cap = cv2.VideoCapture('60_FULL_N_1.avi')
    # # params for ShiTomasi corner detection
    # feature_params = dict( maxCorners = 100,
    #                        qualityLevel = 0.3,
    #                        minDistance = 0,
    #                        blockSize = 10 )
    # # Parameters for lucas kanade optical flow
    # lk_params = dict( winSize  = (15,15),
    #                   maxLevel = 2,
    #                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # # Create some random colors
    # color = np.random.randint(0,255,(100,3))
    # # Take first frame and find corners in it
    # ret, old_frame = cap.read()
    # old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # # Create a mask image for drawing purposes
    # mask = np.zeros_like(old_frame)
    # while(cap.isOpened()):
    #
    #     ret,frame = cap.read()
    #     if len(frame)==0:
    #         break
    #     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     # calculate optical flow
    #     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #     # Select good points
    #     good_new = p1[st==1]
    #     good_old = p0[st==1]
    #     # draw the tracks
    #     for i,(new,old) in enumerate(zip(good_new,good_old)):
    #         a,b = new.ravel()
    #         c,d = old.ravel()
    #         mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #         frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    #     img = cv2.add(frame,mask)
    #     cv2.imshow('frame',img)
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break
    #     # Now update the previous frame and previous points
    #     old_gray = frame_gray.copy()
    #     p0 = good_new.reshape(-1,1,2)
    # cv2.destroyAllWindows()
    # cap.release()

    def plot_mazy(self,mazy_factor,width_bar):
        #normalize it
        temp=[]
        max_diff=max(mazy_factor)
        min_diff =min(mazy_factor)
        for i in mazy_factor:
            temp.append((i-min_diff)/max_diff-min_diff)
        print(temp)
        # fig, ax = mat.subplots()
        index = np.arange(len(temp))
        mat.subplot(2,1,1)
        mat.plot(index,temp,'ro')
        mat.ylabel('relative beat sync')
        mat.xlabel('beat location')
        # opacity = 0.4
        # error_config = {'ecolor': '0.3'}

        # rects1 = ax.bar(index, mazy_factor, width_bar,
        #                 alpha=opacity, color='b',
        #                  error_kw=error_config,
        #                 label='Mazy Percentage')
        # ax.set_xlabel('Dance Routine No')
        # ax.set_ylabel('Scores')
        # ax.legend()
        # fig.tight_layout()

    def plot_show(selfs):
        mat.show(block=True)
