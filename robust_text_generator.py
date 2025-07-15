import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

class RobustTextImageGenerator:
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def create_text_image(self, char, font_size, font_path='arial.ttf', add_noise=True):
        """Create text image using OpenCV only - bypasses PIL completely"""
        # Create white background
        img = np.full((self.img_height, self.img_width), 255, dtype=np.uint8)

        # OpenCV font settings
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = font_size / 20.0  # Scale factor
        thickness = max(1, int(font_scale * 2))
        
        # Get text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(
            char, font_face, font_scale, thickness
        )
        
        # Calculate center position
        x = (self.img_width - text_width) // 2
        y = (self.img_height + text_height) // 2
        
        # Add random variations
        if add_noise:
            x += random.randint(-3, 3)
            y += random.randint(-3, 3)
        
        # Draw BLACK text (color=0) on white background
        cv2.putText(img, char, (x, y), font_face, font_scale, 0, thickness)
        
        # Apply rotation if needed
        if add_noise:
            angle = random.randint(-5, 5)
            if angle != 0:
                center = (self.img_width // 2, self.img_height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, rotation_matrix, (self.img_width, self.img_height), 
                                   borderValue=255)
        
        # Add noise variations
        if add_noise:
            # Random noise
            noise = np.random.normal(0, 5, img.shape)
            img = np.clip(img + noise, 0, 255)
            
            # Random brightness
            brightness = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 255)
        
        return img.astype(np.uint8)