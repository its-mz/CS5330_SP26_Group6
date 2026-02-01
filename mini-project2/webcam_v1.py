import cv2
import numpy as np
import time
import os
import platform

class WebCamApp:
    def __init__(self):
        self.x, self.y = 0, 0
        self.angle = 0
        self.scale = 1.0 
        self.perspective_on = False
        self.isFlip = -1
        self.prevTime = time.time()
        self.fps = 0
        #perspective change
        self.tilth = 0.8
        self.tiltv = 0.5 

        
    def translation(self, frame):
        rows = frame.shape[0]
        cols = frame.shape[1]

        translate = np.float32([[1, 0, self.x], [0, 1, self.y]])
        return cv2.warpAffine(frame, translate, (cols, rows))
    
    def rotation(self, frame):
        rows = frame.shape[0]
        cols = frame.shape[1]
        center = (cols // 2, rows // 2)

        rotate = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        return cv2.warpAffine(frame, rotate, (cols, rows))
    
    def scaling(self, frame):
        rows = frame.shape[0]
        cols = frame.shape[1]
        new_cols = int(cols * self.scale)
        new_rows = int(rows * self.scale)
        
        resize = cv2.resize(frame, (new_cols, new_rows), interpolation = cv2.INTER_LINEAR)
        
        canvas = np.zeros((rows, cols, 3), dtype = np.uint8)
        
        offsetX = (cols - new_cols) // 2
        offsetY = (rows - new_rows) // 2
        
        x1scr = max(0, -offsetX)
        y1scr = max(0, -offsetY)
        x2src = min(new_cols, cols - offsetX)
        y2src = min(new_rows, rows - offsetY)
        
        dst_x1 = max(0, offsetX)
        dst_y1 = max(0, offsetY)
        dst_x2 = dst_x1 + (x2src - x1scr)
        dst_y2 = dst_y1 + (y2src - y1scr)
        
        if x2src > x1scr and y2src > y1scr:
            canvas[dst_y1 : dst_y2, dst_x1 : dst_x2] = resize[y1scr : y2src, x1scr : x2src]
        
        return canvas
    
    def perspective(self, frame):
        rows = frame.shape[0]
        cols = frame.shape[1]
        c = cols - 1
        r = rows - 1

        dx = self.tiltv * cols * 0.22
        dy = self.tilth * rows * 0.22

        src = np.float32([[0, 0], [c, 0], [0, r],[c, r]])

        dst = np.float32([[0 + dx, 0 + dy], [c - dx, 0 - dy], [0 - dx,r - dy], [c + dx, r + dy]])

        dst[:, 0] = np.clip(dst[:, 0], 0, c)
        dst[:, 1] = np.clip(dst[:, 1], 0, r)

        H = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, H, (cols, rows))

    
    def flip(self, frame):
        if self.isFlip == 0:
            return cv2.flip(frame, 0)
        elif self.isFlip == 1:
            return cv2.flip(frame, 1)
        else:
            return frame

    
    def process_frame(self, frame):
        res = frame.copy()

        if self.isFlip != -1:
            res = self.flip(res)
        
        if self.x != 0 or self.y != 0:
            res = self.translation(res)
        
        if self.angle != 0:
            res = self.rotation(res)
        
        if self.scale != 1.0:
            res =  self.scaling(res)
        
        if self.perspective_on:
            res =  self.perspective(res)
        
        return res
    
    def calculate_fps(self):
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prevTime + 0.000001)
        self.prevTime = curr_time
        return self.fps
    
    def reset(self):
        self.x = 0
        self.y = 0 
        self.angle = 0
        self.scale = 1.0 
        self.perspective_on = False
        self.isFlip = -1
        self.tilth = 0.8
        self.tiltv = 0.5 



def main():

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Can't open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    app = WebCamApp()
    
    while True:
        
        ret, frame = cap.read() 
        transformed = app.process_frame(frame)
        fps = app.calculate_fps()
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 255, 20), 2)
        cv2.putText(transformed, f"FPS: {fps:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,255, 0), 2)
        
        combined = cv2.hconcat([frame, transformed])
        cv2.imshow("WebCam app", combined)
        
        # debugging arrow key in different OS
        #key = cv2.waitKeyEx(1)
        # if key != -1:
        #     print(f"Key pressed: {key}")
        
        # arrow code different in windows and mac
        # up = [2490368, 65362]
        # down = [2621440, 65364]
        # left = [2424832, 65361]
        # right = [2555904, 65363]
        
        current_os = platform.system() 
        # macOS
        if current_os == "Darwin":  
            up = [63232]
            down = [63233]
            left = [63234]
            right = [63235]
        # Windows/Linux
        else:  
            up = [2490368, 65362]
            down = [2621440, 65364]
            left = [2424832, 65361]
            right = [2555904, 65363]

        key = cv2.waitKeyEx(1)
        
        # if key != -1:
        #     print(key)
        
        if key == ord('q'):
            break
        elif key ==ord('0'):
            app.reset()
            print("rested")
        
        # flip
        elif key == ord('1'):
            if app.isFlip == 0:
                app.isFlip = -1
            else:
                app.isFlip = 0

        elif key == ord('2'):
            if app.isFlip == 1:
                app.isFlip = -1
            else:
                app.isFlip = 1

        # rotation 
        elif key == ord('r'):
            app.angle += 5
        elif key == ord('t'):
            app.angle -= 5
        
        # scale
        elif key == ord('+') or key == ord('='):
            app.scale = min(3.0, app.scale + 0.05)
        elif key == ord('-'):
            app.scale = max(0.1, app.scale - 0.05)
        
        # perspective change
        elif key == ord('p'):
            app.perspective_on = not app.perspective_on
        
        #save 
        elif key == ord('s'):
            output_dir = "output"
            output_path = os.path.join(os.path.dirname(__file__), output_dir)

            # flip naming 
            if app.isFlip == 1:
                sFlip = "horizontal"
            elif app.isFlip == 0:
                sFlip = "vertical"
            else:
                sFlip = "none"
            # perspective naming
            if app.perspective_on:
                sP = "perspective"
            else:
                sP = "no_perspective"

            filename = f"screenshot_{sFlip}_angle{app.angle}_scale{app.scale:.2f}_{sP}.png"
            filename = os.path.join(output_path, filename)
            cv2.imwrite(filename, combined)
        
        # arrow key here 
        elif key in up:
            app.y -= 10
        elif key in down:
            app.y += 10
        elif key in left:
            app.x -= 10
        elif key in right:
            app.x += 10

        # change perspective onee perspective is on
        elif app.perspective_on and key == ord('w'):
            app.tiltv = app.tiltv - 0.05
        elif app.perspective_on and key == ord('x'):
            app.tiltv = app.tiltv + 0.05
        elif app.perspective_on and key == ord('a'):
            app.tilth = app.tilth - 0.05
        elif app.perspective_on and key == ord('d'):
            app.tilth =  app.tilth + 0.05

    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()