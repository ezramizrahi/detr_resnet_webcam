import cv2
import time
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import hashlib

# Initialize the processor and model for object detection
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

def run_object_detection(image):
    # Process the image and generate model inputs
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Convert outputs (bounding boxes and class logits) to a format we can use
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Create a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image, 'RGBA')
    class_colors = {}

    # Functions to generate a color and create a semi-transparent version of the color
    def color_from_label(label):
        hash_object = hashlib.md5(label.encode())
        hash_digest = hash_object.hexdigest()
        return tuple(int(hash_digest[i:i+2], 16) for i in (0, 2, 4))  # RGB

    def transparent_color(color, alpha=20):  # alpha: 0 (transparent) to 255 (opaque)
        return color + (alpha,)

    # Draw bounding boxes and labels on the image
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]

        print(
            f"Detected {label_name} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

        # Assign a color based on a hash of the label
        if label_name not in class_colors:
            class_colors[label_name] = color_from_label(label_name)

        # Draw the box with a semi-transparent fill and thinner border
        color = class_colors[label_name]
        draw.rectangle(box, outline=color, fill=transparent_color(color), width=1)

    return draw_image

def capture_and_process_images(interval, save_path):
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                # Convert the captured frame to PIL image format
                color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(color_converted)

                # Run object detection on the in-memory image
                processed_image = run_object_detection(pil_image)

                # Save the processed image
                timestamp = int(time.time())
                output_image_path = f"{save_path}/IMG_{timestamp}.jpg"
                processed_image.save(output_image_path)
                print(f"Processed image saved as {output_image_path}")

                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Wait for 'interval' seconds
                time.sleep(interval)
            else:
                print("Error: Cannot read frame")
                break
    except KeyboardInterrupt:
        print("Interrupted by user, exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Capture an image every 5 seconds and process it
capture_and_process_images(5, "./output")