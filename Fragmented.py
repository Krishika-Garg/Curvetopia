import cv2
import numpy as np

def regularize_curve(curve, epsilon=0.001):
    """
    Regularize a curve by removing unnecessary points.
    Simplifies the contour based on epsilon.
    """
    approx_curve = cv2.approxPolyDP(curve, epsilon, True)
    return approx_curve

def smooth_curve(curve, size=5):
    """
    Smooth a curve using a moving average filter.
    """
    curve = np.array(curve, dtype=np.float32)
    kernel = np.ones((size, 1)) / size
    curve_smooth = cv2.filter2D(curve, -1, kernel)
    return curve_smooth

def main():
    # Load the image
    img = cv2.imread('image_path')

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to segment the contour
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours
    for contour in contours:
        # Regularize the contour
        contour_reg = regularize_curve(contour)

        # Smooth the contour
        contour_smooth = smooth_curve(contour_reg)

        # Draw the original contour
        cv2.drawContours(img, [contour], -1, (0, 0, 255), 1)

        # Draw the regularized contour
        cv2.drawContours(img, [contour_reg], -1, (0, 255, 0), 1)

        # Draw the smoothed contour
        cv2.drawContours(img, [contour_smooth.astype(np.int32)], -1, (255, 0, 0), 1)

    # Display the result
    cv2.imshow('Contour Smoothing', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
