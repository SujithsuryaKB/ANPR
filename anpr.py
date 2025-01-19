
import cv2
import pytesseract

# Configure Tesseract OCR executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def anpr(image_path):
    """
    Perform ANPR on the given image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: Extracted license plate text.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection using Canny
    edges = cv2.Canny(filtered, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    license_plate = None
    for contour in contours:
        # Approximate the contour
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour has 4 sides (possible rectangle)
        if len(approx) == 4:
            license_plate = approx
            break

    if license_plate is None:
        print("License plate not detected.")
        return ""

    # Create a mask for the license plate
    mask = cv2.drawContours(image, [license_plate], -1, (255, 255, 255), -1)
    out = cv2.bitwise_and(image, image, mask=mask)

    # Crop the license plate region
    (x, y, w, h) = cv2.boundingRect(license_plate)
    cropped = gray[y:y+h, x:x+w]

    # Use Tesseract OCR to extract text
    text = pytesseract.image_to_string(cropped, config='--psm 8')
    print("License Plate Text:", text.strip())

    # Display the images for debugging (optional)
    cv2.imshow("Original Image", image)
    cv2.imshow("Grayscale Image", gray)
    cv2.imshow("Cropped License Plate", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return text.strip()

# Example usage
if __name__ == "__main__":
    image_path = "car_image.jpg"  # Replace with the path to your image
    license_plate_text = anpr(image_path)
