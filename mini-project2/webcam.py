import cv2
import numpy as np
import time
import os

class WebCamApp:
    def __init__(self):
        self.tx, self.ty = 0, 0
        self.angle = 0
        self.scale = 1.0 
        self.perspective_on = False
        self.flip_mode = -2
        self.prev_time = time.time()
        self.fps = 0
        
    def translation(self, frame):
        rows, cols = frame.shape[:2]
        M = np.float32([[1, 0, self.tx], 
                        [0, 1, self.ty]])
        return cv2.warpAffine(frame, M, (cols, rows))
    
    def rotation(self, frame):
        rows, cols = frame.shape[:2]
        center = (cols // 2, rows // 2)

        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        return cv2.warpAffine(frame, M, (cols, rows))
    
    def scaling(self, frame):
        rows, cols = frame.shape[:2]
        new_cols = int(cols * self.scale)
        new_rows = int(rows * self.scale)
        
        resized = cv2.resize(frame, (new_cols, new_rows), interpolation = cv2.INTER_LINEAR)
        
        canvas = np.zeros((rows, cols, 3), dtype = np.uint8)
        
        x_offset = (cols - new_cols) // 2
        y_offset = (rows - new_rows) // 2
        
        src_x1 = max(0, -x_offset)
        src_y1 = max(0, -y_offset)
        src_x2 = min(new_cols, cols - x_offset)
        src_y2 = min(new_rows, rows - y_offset)
        
        dst_x1 = max(0, x_offset)
        dst_y1 = max(0, y_offset)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]
        
        return canvas
    
    def perspective(self, frame):
        rows, cols = frame.shape[:2]
        
        src_pts = np.float32([
            [0, 0],
            [cols, 0],
            [0, rows],
            [cols, rows]
        ])
        
        dst_pts = np.float32([
            [cols * 0.1, rows * 0.1],
            [cols * 0.9, rows * 0.05],
            [cols * 0.05, rows * 0.95],
            [cols * 0.95, rows * 0.9]
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(frame, M, (cols, rows))
    
    def flip(self, frame):
        if self.flip_mode == 1:
            return cv2.flip(frame, 1)
        elif self.flip_mode == 0:
            return cv2.flip(frame, 0)
        return frame
    
    def process_frame(self, frame):
        result = frame.copy()
        
        if self.flip_mode != -2:
            result = self.flip(result)
        
        if self.tx != 0 or self.ty != 0:
            result = self.translation(result)
        
        if self.angle != 0:
            result = self.rotation(result)
        
        if self.scale != 1.0:
            result = self.scaling(result)
        
        if self.perspective_on:
            result = self.perspective(result)
        
        return result
    
    def calculate_fps(self):
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time + 1e-6)
        self.prev_time = curr_time
        return self.fps
    
    def get_status_text(self):
        flip_str = "H" if self.flip_mode == 1 else ("V" if self.flip_mode == 0 else "None")
        return (f"FLIP = {flip_str} | tx={self.tx}, ty={self.ty} | "
                f"angle = {self.angle} | scale = {self.scale:.2f} | persp = {self.perspective_on}")
    
    def reset(self):
        self.tx, self.ty = 0, 0
        self.angle = 0
        self.scale = 1.0
        self.perspective_on = False
        self.flip_mode = -2

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    app = WebCamApp()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break
        
        transformed = app.process_frame(frame)
        
        fps = app.calculate_fps()
        
        cv2.putText(frame, f"Original | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(transformed, app.get_status_text(), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(transformed, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        combined = cv2.hconcat([frame, transformed])
        
        cv2.imshow("WebCam App - Original | Transformed", combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('0'):
            app.reset()
            print("Reset all transformations")
        elif key == ord('1'):
            app.flip_mode = 1 if app.flip_mode != 1 else -2
            print(f"Horizontal flip: {'ON' if app.flip_mode == 1 else 'OFF'}")
        elif key == ord('2'):
            app.flip_mode = 0 if app.flip_mode != 0 else -2
            print(f"Vertical flip: {'ON' if app.flip_mode == 0 else 'OFF'}")
        elif key == ord('r'):
            app.angle += 5
            print(f"Rotation: {app.angle}°")
        elif key == ord('t'):
            app.angle -= 5
            print(f"Rotation: {app.angle}°")
        elif key == ord('+') or key == ord('='):
            app.scale = min(3.0, app.scale + 0.05)
            print(f"Scale: {app.scale:.2f}")
        elif key == ord('-'):
            app.scale = max(0.1, app.scale - 0.05)
            print(f"Scale: {app.scale:.2f}")
        elif key == ord('p'):
            app.perspective_on = not app.perspective_on
            print(f"Perspective: {'ON' if app.perspective_on else 'OFF'}")
        elif key == ord('h'):
            app.print_help()
        elif key == ord('s'):
            output_dir = "output"
            output_path = os.path.join(os.path.dirname(__file__), output_dir)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                
            #filename = os.path.join(output_path,f"screenshot_{int(time.time())}.png")
            
            flip_str = "horizontal" if app.flip_mode == 1 else "vertical" if app.flip_mode == 0 else "none"
            perspective_str = "perspective" if app.perspective_on else "no_perspective"
            filename = f"screenshot_{flip_str}_tx{app.tx}_ty{app.ty}_angle{app.angle}_scale{app.scale:.2f}_{perspective_str}_{int(time.time())}.png"
            filename = os.path.join(output_path, filename)
    
            cv2.imwrite(filename, combined)
            print(f"Screenshot saved: {filename}")
        elif key == 82 or key == 0:
            app.ty -= 10
        elif key == 84 or key == 1:
            app.ty += 10
        elif key == 81 or key == 2:
            app.tx -= 10
        elif key == 83 or key == 3:
            app.tx += 10
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()