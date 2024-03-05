import os
import scipy.io
import numpy as np
import sys 
sys.path.append('..')
from utils.eval import *

r'''This script is dedicated to choose the top five scored results in.
'''

def find_best_5(folder_path):
    # Initialize a list to store file names and corresponding scores
    scores_list = []

    # Loop through the folders and find .mat files
    for folder_name in os.listdir(folder_path):
        folder_dir = os.path.join(folder_path, folder_name)

        if os.path.isdir(folder_dir):
            # Find .mat files in the folder
            mat_files = [file for file in os.listdir(folder_dir) if file.endswith(".mat")]

            if len(mat_files) == 1:
                # Load the .mat file
                mat_file_path = os.path.join(folder_dir, mat_files[0])
                mat_data = scipy.io.loadmat(mat_file_path)

                # Extract the score (replace 'score' with the actual field name in your .mat files)
                score = rmse_based_numpy(mat_data['SSH_ref'],mat_data['SSH_est'][:600,:600])
                

                # Store the score and the corresponding file path in the list
                scores_list.append((mat_file_path, score))

    # Sort the files based on their scores and select the top 5
    sorted_files = sorted(scores_list, key=lambda x: x[1], reverse=True)
    best_5_scores = sorted_files[:5]

    return best_5_scores


def main():
    folder_path = "../results/Satellite/CV_TV_V/2012-10-13"
    best_5_scores = [elements[0] for elements in find_best_5(folder_path)]
    for score in best_5_scores:
        print(score)


if __name__=='__main__':
    main()