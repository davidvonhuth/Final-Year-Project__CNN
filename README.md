Final Year Project 2021; ECS771U; BSc Computer Science; Queen Mary, University of London

Student name:		David von Huth
Project Title:  	"Micro-event detection in EEG signals using Convolutional Neural Networks"

This text file is meant to accompany the pre-processing script and CNN training script that can be found in the same zipped folder. 




--------------------------------------------------< INSTRUCTIONS PRE-PROCESSING SCRIPT >----------------------------------------------------

In order to make the training script "pre-processing.py" run, the raw PSG files must first be downloaded. For this project, the data
was taken from the Montreal Archive of Sleep Studies (MASS), cohort 1, subset SS2. From this source, the PSGs and K-complex 
annotations come as EDF files, while the sleep spindle annotations come as csv files. They should each be stored in their own folder;
"PSG_files", "KC_labels" and "SS_labels" respectively. From this point, the following instructions should be followed:

1. 	Place the three folders "PSG_files", "KC_labels" and "SS_labels" inside the same zipped folder where "pre-processing.py" and 
	"cnn_training.ipynb" exists.
2.	Ensure Python 3 is installed on the computer used.
3.	Ensure all required libraries are pre-installed on the computer (look at all imported libraries on the top of "pre-processing.py"). 
	Here's a link to install pyEDFlib (the main library used for reading EDF files): https://pypi.org/project/pyEDFlib/
4.	Run the following command: python3 pre-processing.py
5.	The script will now create a pickled file, storing the processed data. The filename will by default be: "pre_2-sec_0.5_EEG_data.pickle"
	because it assumes a segment size of 2 seconds, and a threshold value of 0.5 seconds. The threshold value reflects the minimum
	percentage duration of a micro-event needed to label a segment as said micro-event. For example, if the threshold is 0.4 (40%) for
	a given sleep spindle, it means at least 40% of the sleep spindle must fall within a segment in order to label that segment
	a sleep spindle. If this criteria is not met, the segment will be labelled as "Other".
6.	The pickled file should be copied to the storage location from which Neural Network training may be enabled. Many different platforms
	may be used with their own configurations, but this project used Google Drive for storing the pickled file. Store the pickled file
	in a directory with some suitable name. 





---------------------------------------------------< INSTRUCTIONS CNN TRAINING SCRIPT >-----------------------------------------------------

Once the pickled file has been uploaded to Google Drive, the following instructions should be followed:

1.	Upload the training script "cnn_training.ipynb" to the same folder as the pickled file. 
2.	Inside the training script, ensure that the pickle-deserialising code block has the correct file path to your personal directory. 
	For example:
	filepath = '/content/drive/My Drive/CNN/pre3_2-sec_0.5_EEG_data.pickle'
	with open(filepath, 'rb') as f:
  		X = pickle.load(f) 
  		y = pickle.load(f) 
3.	Further down in the script, in the code cell under "Automatic threshold-setting", ensure "model_filepath" is set to a Google Drive
	directory where weights will be stored for well performing epochs. 
4. 	In the code cell under "Time saving F1 calculator", ensure that the string values in "epoch_list" correspond to epochs stored in 
	"model_filepath". In the same code cell, also ensure that "model_filepath" is set to the same path as the one from point 3 above, with
	the correct filename. 
5.	Run all the cells from top to bottom, ensuring Google Drive is mounted properly by following the instructions for it once prompted. 
5.	After running the cell under "Time saving F1 calculator", then F1-score, Precision and Recall results should be displayed 
	per chosen epoch for the model defined.
