import cv2
import easyocr
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumberPlateDetector:
    def __init__(self):
        try:
            # Get base directory and create paths
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.models_dir = os.path.join(base_dir, "models")
            self.cascade_path = os.path.join(base_dir, "resources", "numberplate_haarcade.xml")

            # Create models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)

            # Check if cascade file exists
            if not os.path.exists(self.cascade_path):
                raise FileNotFoundError(f"Cascade file not found at {self.cascade_path}")

            # Initialize cascade classifier
            self.detector = cv2.CascadeClassifier(self.cascade_path)
            if self.detector.empty():
                raise ValueError("Failed to load cascade classifier")

            # Check if model files exist
            model_files = {
                'craft_mlt_25k.pth': os.path.join(self.models_dir, 'craft_mlt_25k.pth'),
                'english_g2.pth': os.path.join(self.models_dir, 'english_g2.pth')
            }

            # Initialize EasyOCR based on model availability
            download_needed = not all(os.path.exists(path) for path in model_files.values())

            logger.info(f"Initializing EasyOCR with models from: {self.models_dir}")
            logger.info(f"Download needed: {download_needed}")

            # Initialize EasyOCR with appropriate settings
            self.reader = easyocr.Reader(
                ['en'],
                model_storage_directory=self.models_dir,
                download_enabled=download_needed,  # Only download if models don't exist
                recog_network='english_g2',
                gpu=False  # Set to True if GPU is available
            )

            logger.info("NumberPlateDetector initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing NumberPlateDetector: {str(e)}")
            raise

    def detect_plate(self, img):
        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect Number plates
            plates = self.detector.detectMultiScale(
                img_gray, scaleFactor=1.05, minNeighbors=7,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            detected_plates = []
            logger.info(f"Found {len(plates)} potential plates")

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
                logger.info(f"Detected plate text: {plate_text}")

                # draw text in the frame
                cv2.putText(img, plate_text, (x, y - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

            return img, detected_plates

        except Exception as e:
            logger.error(f"Error in detect_plate: {str(e)}")
            raise


def process_image(input_image_path: str, output_dir: str = "output_images"):
    try:
        logger.info(f"Processing image: {input_image_path}")

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

        logger.info(f"Processing completed. Detected plates: {detected_plates}")
        return output_path, detected_plates

    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
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