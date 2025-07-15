import easyocr

class MenuTokenExtractor:
    def __init__(self, low_text=0.3, width_ths=0.4, height_ths=0.4):
        self.low_text = low_text
        self.width_ths = width_ths
        self.height_ths = height_ths
        self.reader = easyocr.Reader(['en', 'fr'], gpu=False, verbose=False)

    def extract_tokens(self, image_path):

      """Simple pipeline: extract words with x,y positions"""
      
      # Initialize EasyOCR
      
      # Extract text with positions
      results = self.reader.readtext(
         image_path, 
         detail=1, 
         paragraph=False, 
         low_text=self.low_text, 
         width_ths=self.width_ths, 
         height_ths=self.height_ths, 
         blocklist=[";!@#^&*()_+=`~\"<>?/:;{}[]|\\"]  # Exclude special characters
        )
      
      # Convert to simple format
      text_data = []
      for detection in results:
        bbox = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text = detection[1]
        confidence = detection[2]
        
        # Calculate center position
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x = int(sum(x_coords) / 4)  # Center x
        y = int(sum(y_coords) / 4)  # Center y
        
        # width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)

        text_data.append({
            'text': text,
            'x': x,
            'y': y,
            'h' : height.item(),
            'confidence': confidence.item()
        })
      
      return text_data


