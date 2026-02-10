import cv2
import numpy as np
import time

def initialize_video_capture(mode = '2-camera'):
    cap_l = cv2.VideoCapture(0)
    
    if mode == '2-camera':
        cap_r = cv2.VideoCapture(1)
        
        if not cap_r.isOpened():
            print("Second camera not found.")
            cap_l.release()
            return None, None, None, mode
        return cap_l, cap_r, None, mode
    
    else: 
        static_img = cv2.imread('data/vita1.jpg')
        
        if static_img is None:
            print("Error: logo1.jpg missing for simulation.")
            cap_l.release()
            return None, None, None, mode
        
        return cap_l, None, static_img, mode
    
def initialize_sift_detector():
    sift = cv2.SIFT_create(nfeatures=500)
    return sift

def initialize_flann_matcher():
    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann

def frame_processor(cap_l, cap_r, static_img, simulation_mode, sift, flann):
    while True:
        start_time = time.time()
        
        if mode == '2-camera':
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r:
                break
        else:
            ret_l, frame_l = cap_l.read()
            if not ret_l:
                break
            frame_r = cv2.resize(static_img, (frame_l.shape[1], frame_l.shape[0]))

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(gray_l, None)
        kp2, des2 = sift.detectAndCompute(gray_r, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                    cv2.imshow('Sewing Machine', frame_l)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
        
        matches = flann.knnMatch(des1, des2, k = 2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        matched_img = cv2.drawMatches(
            frame_l, 
            kp1, 
            frame_r, 
            kp2, 
            good_matches, 
            None, 
            flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow('Matched Features', matched_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
if __name__ == "__main__":
    mode = '1-camera'

    cap_l, cap_r, static_img, mode = initialize_video_capture(mode)

    sift = initialize_sift_detector()
    flann = initialize_flann_matcher()

    frame_processor(cap_l, cap_r, static_img, mode, sift, flann)
    
    if mode == '2-camera':
        cap_l.release()
        cap_r.release()
    else:
        cap_l.release()
        
    cv2.destroyAllWindows()