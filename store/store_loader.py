import glob
from config.wine import RED_VARIETALS, WHITE_VARIETALS, COUNTRIES
import re 
import pandas as pd

RED_REGEX = re.compile(r'\b(' + '|'.join(re.escape(varietal.lower()) for varietal in RED_VARIETALS) + r')\b', re.IGNORECASE)
WHITE_REGEX = re.compile(r'\b(' + '|'.join(re.escape(varietal.lower()) for varietal in WHITE_VARIETALS) + r')\b', re.IGNORECASE)
YEAR_REGEX = re.compile(r'\b(\d{4})\b')

COUNTRY_PATTERNS = {pattern.lower(): country.lower() for country, data in COUNTRIES.items() for pattern in data['patterns']}

WINE_DTYPES = {
  'country'     : 'category',      # categorical for repeated values
  'region'      : 'string',
  'year'        : 'Int16',
  'rank'        : 'Float32',
  'winery'      : 'string',
  'description' : 'string',
  'type'        : 'category',      # categorical for repeated values
  'variety'     : 'category', 
  'price'       : 'Float32',       # float32 for currency precision
}


def extract_country(text):
    if not text:
        return
    
    text = text.strip().lower()
    
    # ISO search
    if text in COUNTRIES.keys():
        return text
    
    # pattern match
    matches = [country for pattern, country in COUNTRY_PATTERNS.items() if text.startswith(pattern)]

    if any(matches): 
        return matches[0]
    
def extract_year(text):
    if not text:
        return
    
    match = YEAR_REGEX.search(str(text))
    if match:
        return int(match.group(1))
    return


def extract_varietal(text):
    """
    Extract wine type and varietal from text.
    
    Args:
        text (str): Input text to search for wine varietals
        
    Returns:
        tuple: (type, varietal) where type is 'red'/'white' and varietal is the specific variety,
              or (None, None) if no varietal found
    """
    if not text:
        return None, None
    
    text_str = str(text)
    
    # Search for red varietals
    red_match = RED_REGEX.search(text_str)
    if red_match:
        # Find the original varietal name (with proper casing)
        matched_varietal = red_match.group(1).lower()
        for varietal in RED_VARIETALS:
            if varietal.lower() == matched_varietal:
                return 'red', varietal
    
    # Search for white varietals
    white_match = WHITE_REGEX.search(text_str)
    if white_match:
        # Find the original varietal name (with proper casing)
        matched_varietal = white_match.group(1).lower()
        for varietal in WHITE_VARIETALS:
            if varietal.lower() == matched_varietal:
                return 'white', varietal
    
    # No varietal found
    return None, None

class StoreLoader:
  """
  Base class for loading wine data.
  """

  def __init__(self, path_pattern):
    self.path_pattern = path_pattern

  def load(self):
    """
    Load and process wine data.
    """
    files = glob.glob(self.path_pattern, recursive=True)
    if len(files) == 0:
        raise ValueError(f"No files found matching the pattern {self.path_pattern}")
    
    df = pd.read_csv(files[0])

    for file in files[1:]:
      df = pd.merge(df, pd.read_csv(file), how='outer')

    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.strip()

    return df

