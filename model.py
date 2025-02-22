import cv2
import easyocr
import os


class NumberPlateDetector:
    def __init__(self):
        # Get the absolute path to the resources directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(base_dir, "resources", "numberplate_haarcade.xml")

        # Initialize cascade classifier
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise ValueError(f"Error: Cascade classifier could not be loaded from {cascade_path}")

        # Initialize the easyocr Reader object
        self.reader = easyocr.Reader(['en'])

    def detect_plate(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect Number plates
        plates = self.detector.detectMultiScale(
            img_gray, scaleFactor=1.05, minNeighbors=7,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        detected_plates = []

        # iterate through each detected number plate
        for (x, y, w, h) in plates:
            # draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the numberplate
            plateROI = img_gray[y:y + h, x:x + w]

            # detect text
            text = self.reader.readtext(plateROI)

            if len(text) == 0:
                continue

            plate_text = text[0][1]
            detected_plates.append(plate_text)

            # draw text in the frame
            cv2.putText(img, plate_text, (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        return img, detected_plates


def process_image(input_image_path: str, output_dir: str = "output_images"):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize the detector
        detector = NumberPlateDetector()

        # Read the input image
        img = cv2.imread(input_image_path)
        if img is None:
            raise ValueError(f"Could not read image at {input_image_path}")

        # Process the image
        processed_img, detected_plates = detector.detect_plate(img)

        # Save the output image
        output_path = os.path.join(output_dir, 'output_image.jpg')
        cv2.imwrite(output_path, processed_img)

        return output_path, detected_plates

    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        raise


# Example usage for testing
if __name__ == "__main__":
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_image = os.path.join(base_dir, "resources", "inputimage.jpeg")
        output_path, detected_plates = process_image(test_image)
        print(f"Image processing completed. Output saved to: {output_path}")
        print(f"Detected plates: {detected_plates}")
    except Exception as e:
        print(f"Error during testing: {str(e)}")