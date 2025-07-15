import re

def _reverse_country_patterns(wine_regions):
   """Create reverse lookup dictionary from pattern/country name to ISO code"""
   return {pattern.lower(): iso_code 
           for iso_code, country_data in wine_regions.items() 
           for pattern in country_data['patterns']}

def _reverse_country_regions(wine_regions):
   """Create reverse lookup dictionary from wine region to ISO code"""
   return {region.lower(): iso_code 
           for iso_code, country_data in wine_regions.items() 
           for region in country_data['regions']}

class MenuTokenEnrichor:
  def __init__(self, red_varietals, white_varietals, varietal_abbreviations, countries):
    self.red_varietals = red_varietals
    self.white_varietals = white_varietals 
    self.varietal_abbreviations = varietal_abbreviations
    self.countries = [c.lower() for c in countries.keys()]
    self.country_regions = _reverse_country_regions(countries)
    self.country_patterns = _reverse_country_patterns(countries)

  def enrich_token(self, token):
    text = token['text'].strip()

    # Country
    if text.lower() in self.countries:
      token['is_country'] = True
      token['text'] = text.lower()
      return token

    elif text.lower() in self.country_patterns:
      token['is_country'] = True
      token['text'] = text.lower()
      return token

    # Region
    if text.lower() in self.country_regions:
      token['is_region'] = True
      token['text'] = text.lower()

    current_year = 2025  # Replace with dynamic year if needed
    vintage_pattern = rf'\b(19[5-9]\d|20[0-{str(current_year)[-1]}]\d)(?:\s*(?:vintage|v\.?))?(?:\s*wine)?\b'

    if re.search(vintage_pattern, text, re.IGNORECASE):
      token['is_vintage'] = True
      token['text'] = text.replace('O', '0').replace('I', '1').replace('l', '1').strip()
      return token

    if text.lower() in self.varietal_abbreviations:
      text = self.varietal_abbreviations[text.lower()]
      token['is_varietal_other'] = True

    if text in self.red_varietals:
      token['is_varietal_red'] = True
      token['text'] = text
      return token

    elif text in self.white_varietals:
      token['is_varietal_white'] = True
      token['text'] = text
      return token

    if text.isdigit() or text.startswith('$') or text.startswith('€') or text.startswith('£') or text.endswith('$') or text.endswith('€') or text.endswith('£'):
      token['is_price'] = True
      token['text'] = ''.join([char for char in text if char.isdigit()])
      return token

    return token
