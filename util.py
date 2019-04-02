import cv2
import numpy as np

def preprocess_frame(frame):
    """
    Turn the frame into gray,
    resize it and crop to 84,84
    """
    #preprocess image
    preprocessed_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    preprocessed_image = cv2.resize(preprocessed_image, (84, 110))
    preprocessed_image = preprocessed_image[26:, :]
    return preprocessed_image

def stack_frames(stacked_frames, state, is_new_episode, frame_stack_size=4):
    """
    If it is a new episode, stack the given state <frame_stack_size> times,
    else stack the current state
    """
    frame = preprocess_frame(state)
    if is_new_episode:
        for i in range(frame_stack_size):
            if(len(stacked_frames) >= frame_stack_size):
                stacked_frames = np.delete(stacked_frames, 0, axis=0)
            stacked_frames = np.append(stacked_frames, np.expand_dims(frame, axis=0), axis=0)
    else:
        if(len(stacked_frames) >= frame_stack_size):
            stacked_frames = np.delete(stacked_frames, 0, axis=0)    
        stacked_frames = np.append(stacked_frames, np.expand_dims(frame, axis=0), axis=0)
    return stacked_frames
