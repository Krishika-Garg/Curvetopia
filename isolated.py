import cv2
import numpy as np
from skimage.morphology import skeletonize

def thinning(src):
    """Perform thinning using skimage's skeletonize."""
    src = src // 255  # Convert to binary image
    skeleton = skeletonize(src).astype(np.uint8) * 255
    return skeleton

def detect_shape(contour):
    """Classify shape based on contour properties."""
    if len(contour) >= 10:
        # Check if it might be a star shape (polygon with 10 vertices)
        hull = cv2.convexHull(contour)
        if len(hull) == 10:
            return "Star"

    # Compute convex hull
    hull = cv2.convexHull(contour)

    # Compute circularity
    area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(hull, True)
    if perimeter == 0:
        return "Unknown"
    circularity = (4 * np.pi * area) / (perimeter * perimeter)

    if circularity > 0.9:
        return "Circle"
    elif circularity > 0.75:
        return "Rectangle"
    elif circularity > 0.7:
        return "Triangle"
    else:
        return "Other"

def draw_straight_lines(img, bin_img):
    """Detect and draw straight lines using Hough Transform."""
    edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            #cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw line in green
    return img

def draw_contour_lines(img, contour, shape):
    """Draw lines between the vertices of the contour, unless the shape is a circle."""
    if shape == "Circle":
        # Fit a circle to the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(img, center, radius, (0, 0, 255), 2)  # Draw circle in red
    else:
        # Approximate the contour to get vertices
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Draw lines between adjacent vertices
        for i in range(len(approx)):
            start_point = tuple(approx[i][0])
            end_point = tuple(approx[(i + 1) % len(approx)][0])
            cv2.line(img, start_point, end_point, (0, 0, 255), 2)  # Draw line in red

    return img

def main():
    # Load image
    src = cv2.imread("image_path")  # Update with your image path

    # Convert to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Binarize
    _, bin_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Perform thinning
    bin_img = thinning(bin_img)

    # Create result image
    res = src.copy()

    # Draw straight lines
    res = draw_straight_lines(res, bin_img)

    # Find contours
    contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Classify shape
        shape = detect_shape(contour)

        # Draw contour lines
        res = draw_contour_lines(res, contour, shape)

    # Display result
    cv2.imshow("Result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()
