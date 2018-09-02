# Raks_Roll_Thesis
Raks&amp;Roll is an application that serves the purpose of a virtual choreographer, with the ability to intelligently process music given to it as input and generate a well synchronized dance routine presented to the user in the form of a video.
The input audio file is segmented and processed for the extraction of audio features such as tempo, beat timings etc. Different parts of the audio are classified on the basis of these features. Each audio segment is allocated the most suitable dance move from a variety of pre- learned dance steps. These dance moves are then combined to create an output video presented to the user.

Our work so far can produce a dance routine for the BPMs 60, 90 and 120, as the moves were recorded on these. The routine is not very diverse or smooth due to a very small collection of moves. The scope of our project included segmentation of audio to retrieve meaningful information which we have achieved using the audio processing libraries Librosa and Essentia, The major addition to our algorithm (MAZY) in second phase was that we have started considering the local beats tempo in our heuristic function. This ensures that maximum sync in achieved. Moreover we have implemented a website and android application supported by our backend server running on amazon EC2 instance. Android application has a feedback module that tracks the feedback from the user. This feedback is used to list smoothness and beats syncing. This feedback is incorporated into learning module that uses probabilistic model to enhance the selection of moves against songs provided. Moreover we have rebuilt the data set to ensure that dance moves are recorded with a context to enhance smoothness of final dance routine that is generated.

# App Wire frames

<img src="https://github.com/Yahyaali1/Raks_Roll_Thesis/blob/master/Design/Asset%201.png" height="350" width="1100">

# Results

![Alt text](https://github.com/Yahyaali1/Raks_Roll_Thesis/blob/master/Results/result_Sample.gif)

# More Results
<a href="https://www.youtube.com/watch?v=OBTSdGEk0Ng&list=PLoWPKdTHNZ9Te62Lz87CRk80YwV_Eu0Om
" target="_blank"><img src="https://github.com/Yahyaali1/Raks_Roll_Thesis/blob/master/sampleImage.JPG" 
alt="Click to watch the results" width="540" height="450" border="10" /></a>



For future, we intend to use artificial intelligence to gauge the entertainment factor of the dance produced, and also be able to produce dynamic pacing of a move for an extracted BPM that did not exist in the dance data.

# Team
![Alt text](https://media.licdn.com/dms/image/C5112AQF9uUl4DB2nFA/article-inline_image-shrink_1500_2232/0?e=1541030400&v=beta&t=fXIEgoWf4ep313kpK0G02vVcLNcdQqbhHIyJWVXkAro)




