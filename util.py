import cv2
import numpy as np

def preprocess_frame(frame):
    #preprocess image
    preprocessed_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    preprocessed_image = cv2.resize(preprocessed_image, (84, 110))
    preprocessed_image = preprocessed_image[26:, :]
    return preprocessed_image

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        for i in range(4):
            if(len(stacked_frames) >= 4):
                stacked_frames = np.delete(stacked_frames, 0, axis=0)
            stacked_frames = np.append(stacked_frames, np.expand_dims(frame, axis=0), axis=0)
    else:
        if(len(stacked_frames) >= 4):
            stacked_frames = np.delete(stacked_frames, 0, axis=0)    
        stacked_frames = np.append(stacked_frames, np.expand_dims(frame, axis=0), axis=0)
    return stacked_frames
