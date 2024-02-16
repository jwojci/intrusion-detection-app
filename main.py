# Create a background subtractor
# Apply the subtractor to every frame to create a foreground mask
# Apply erosion to the mask to remove noise
# Create contours from the eroded foreground mask
# Find the largest contour
# Find the bounding rectangle
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

source = "intruder_2.mp4"

video_cap = cv2.VideoCapture(source)

if not video_cap.isOpened():
    print('Unable to open: ' + source)

frame_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_cap.get(cv2.CAP_PROP_FPS))

size = (frame_w, frame_h)
frame_area = frame_w * frame_h
video_out_alert_file = 'video_out_alert.mp4'
video_out_alert = cv2.VideoWriter(video_out_alert_file, cv2.VideoWriter_fourcc(*"XVID"), fps, size)


def drawBannerText(frame, text, banner_height_percent=0.08, font_scale=0.8, text_color=(0, 255, 0),
                   font_thickness=2):
    # Draw a black filled banner across the top of the image frame
    # percent: set the banner height as a percentage of the frame height
    banner_height = int(banner_height_percent * frame.shape[0])
    # (0, 0) is the top left corner and (frame.shape[1], banner_height) is the bottom right corner of the rectangle
    # that will be drawn
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_height), (0, 0, 0), thickness=-1)

    # Draw text on banner
    left_offset = 20
    location = (left_offset, int(10 + (banner_height_percent * frame.shape[0]) / 2))
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)


# Create the background subtractor
bg_sub = cv2.createBackgroundSubtractorKNN(history=200)

# Process video for analysis
ksize = (5, 5)  # kernel size for erosion
max_countours = 3  # number of contours to use for rendering a bounding rectangle
frame_count = 0
min_contour_area_thresh = 0.01  # minimum fraction of frame required for maximum contour

yellow = (0, 255, 255)
red = (0, 0, 255)

# Process video frames
while True:
    ret, frame = video_cap.read()
    frame_count += 1
    if frame is None:
        break

    # Stage 1: Create a foreground mask for the current frame
    fg_mask = bg_sub.apply(frame)

    # Stage 2: Stage 1 + Erosion
    fg_mask_erode = cv2.erode(fg_mask, np.ones(ksize, np.uint8))

    # Stage 3: Stage 2 + Contours
    contours_erode, hierarchy = cv2.findContours(fg_mask_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_erode) > 0:
        # Sort contours based on area
        contours_sorted = sorted(contours_erode, key=cv2.contourArea, reverse=True)

        # Contour area of largest contour
        contour_area_max = cv2.contourArea(contours_sorted[0])

        # Compute fraction of total frame area occupied by largest contour
        contour_frac = contour_area_max / frame_area

        # Configrm contour_frac is bigger than min_contour_area_thresh threshold
        if contour_frac > min_contour_area_thresh:
            # Compute bounding rectangle for the top N largest contours
            for idx in range(min(max_countours, len(contours_sorted))):
                xc, yc, wc, hc = cv2.boundingRect(contours_sorted[idx])
                if idx == 0:
                    x1 = xc
                    y1 = yc
                    x2 = xc + wc
                    y2 = yc + hc
                else:
                    x1 = min(x1, xc)
                    y1 = min(y1, yc)
                    x2 = max(x2, xc + wc)
                    y2 = max(y2, yc + hc)

            # Draw bounding rectangle for top N contours on output frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, thickness=2)
            drawBannerText(frame, 'Intrusion Alert', text_color=red)

            # Write alert video to file system
            video_out_alert.write(frame)

video_cap.release()
video_out_alert.release()

# Load output video
clip = VideoFileClip(video_out_alert_file)
clip.ipython_display(width=1000)
