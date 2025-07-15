import numpy as np
from PIL import Image, ImageDraw, ImageFont

class TextGenerator:
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height
        
    def create_text_image(self, char, chr_width, chr_height):
        # Create white background canvas
        img = Image.new('L', (self.img_width, self.img_height), 255)
        draw = ImageDraw.Draw(img)
        
        # Try to load a font with the specified character size
        try:
            font = ImageFont.truetype("arial.ttf", size=min(chr_width, chr_height))
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Get text bounding box to calculate actual text size
        if font:
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            # Account for text baseline offset
            text_offset_y = bbox[1]
        else:
            # Fallback if no font available
            text_width = chr_width
            text_height = chr_height
            text_offset_y = 0
        
        # Center the character on the canvas
        x = (self.img_width - text_width) // 2
        y = (self.img_height - text_height) // 2 - text_offset_y
        
        # Draw black text on white background
        draw.text((x, y), char, fill=0, font=font)
        
        # Convert to numpy array
        return np.array(img)


