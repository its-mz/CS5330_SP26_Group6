import cv2
import numpy as np


def region_of_interest(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)

    polygon = np.array([[
        (int(w * 0.1), h), 
        (int(w * 0.9), h), 
        (int(w * 0.65), int(h * 0.35)),
        (int(w * 0.35), int(h * 0.35))
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

 

def make_line_points(img, line):
    slope, intercept = line
    h = img.shape[0]

    y1 = h
    y2 = int(h*0.6)

    if abs(slope) < 1e-6:
        return None

    x1 = int((y1 -intercept) /slope)
    x2 = int((y2 -intercept) / slope)

    return (x1, y1, x2, y2)


def average_lines(img, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        angle = abs(np.degrees(np.arctan(slope)))

        if angle <= 10 or angle >= 85: 
            continue

        intercept = y1 - slope *x1

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope,intercept))

    lane_lines = []

    if len(left_fit) > 0:
        left_avg = np.mean(left_fit, axis=0)
        left_line = make_line_points(img, left_avg)
        if left_line is not None:
            lane_lines.append(left_line)

    if len(right_fit) > 0:
        right_avg = np.mean(right_fit, axis=0)
        right_line = make_line_points(img, right_avg)
        if right_line is not None:
            lane_lines.append(right_line)

    return lane_lines


def draw_lane_lines(img, lines):
    line_img = np.zeros_like(img)

    for x1, y1, x2, y2 in lines:
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 5)

    result = cv2.addWeighted(img, 0.8, line_img, 1.0, 0)
    return result, line_img


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi/180,
        threshold=30,
        minLineLength=20, 
        maxLineGap=200
    )
    lane_lines = average_lines(frame, lines)
    result, line_img = draw_lane_lines(frame, lane_lines)
    return gray, blur, edges, roi, line_img, result

def main():

    cap = cv2.VideoCapture("road.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray, blur, edges, roi, line_img, result = detect_lanes(frame)

        cv2.imshow("og", frame)
        cv2.imshow("ray", gray)
        cv2.imshow("Blur", blur)
        cv2.imshow("edges", edges)
        cv2.imshow("roi", roi)
        cv2.imshow("Lane Lines Only", line_img)
        cv2.imshow("Final Result", result)

        key = cv2.waitKey(25) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()