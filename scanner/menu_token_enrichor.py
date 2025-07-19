import re

from config.wine import reverse_country_patterns, reverse_country_regions


class MenuTokenEnrichor:
  def __init__(self, red_varietals: list[str], white_varietals: list[str], varietal_abbreviations: dict[str, str], countries):
    self.red_varietals = [v.lower() for v in red_varietals]
    self.white_varietals = [v.lower() for v in white_varietals] 
    self.varietal_abbreviations = {k.lower(): v.lower() for k, v in varietal_abbreviations.items()}
    self.countries = [c.lower() for c in countries.keys()]
    self.country_regions = reverse_country_regions(countries)
    self.country_patterns = reverse_country_patterns(countries)
    self.price_regex = re.compile(r'(?:[\$€£]?\s*)?(\d+(?:[.,]\d{1,2})?)\s*(?:[\$€£])?', re.IGNORECASE)



    # red_sorted = sorted(self.red_varietals, key=len, reverse=True)
    # white_sorted = sorted(self.white_varietals, key=len, reverse=True)
    # regions_sorted = sorted(self.country_regions.keys(), key=len, reverse=True)

    # self.red_regex = re.compile(r'\b(' + '|'.join(re.escape(v) for v in red_sorted) + r')\b', re.IGNORECASE)
    # self.white_regex = re.compile(r'\b(' + '|'.join(re.escape(v) for v in white_sorted) + r')\b', re.IGNORECASE)
    # self.region_regex = re.compile(r'\b(' + '|'.join(re.escape(r) for r in regions_sorted) + r')\b', re.IGNORECASE)


  def enrich_token(self, token):
    text = token['text'].strip().lower()

    # Region
    if text in self.country_regions:
      token['is_region'] = True
      token['text'] = text
      return token

    # Country
    if text in self.countries:
      token['is_country'] = True
      token['text'] = text
      return token

    elif text in self.country_patterns:
      token['is_country'] = True
      token['text'] = text
      return token

    current_year = 2025  # Replace with dynamic year if needed
    vintage_pattern = rf'\b(19[5-9]\d|20[0-{str(current_year)[-1]}]\d)(?:\s*(?:vintage|v\.?))?(?:\s*wine)?\b'

    if re.search(vintage_pattern, text, re.IGNORECASE):
      token['is_vintage'] = True
      token['text'] = text.replace('o', '0').replace('i', '1').replace('l', '1').strip()
      return token

    # Varietal abbreviations first
    if text in self.varietal_abbreviations:
        varietal = self.varietal_abbreviations[text]
        token['text'] = varietal

        # Check if expanded varietal is red or white
        if varietal in self.red_varietals:
            token['is_varietal_red'] = True
            return token
        
        elif varietal in self.white_varietals:
            token['is_varietal_white'] = True
            return token
        
        else:
          token['is_varietal_other'] = True
          return token

    if text in self.red_varietals:
      token['is_varietal_red'] = True
      token['text'] = text
      return token

    elif text in self.white_varietals:
      token['is_varietal_white'] = True
      token['text'] = text
      return token

    price_match = self.price_regex.search(text)
    if price_match:
      token['is_price'] = True
      token['text'] = price_match.group(1)  # Extract just the number
      return token

    # if text.isdigit() or text.startswith('$') or text.startswith('€') or text.startswith('£') or text.endswith('$') or text.endswith('€') or text.endswith('£'):
    #   price = ''.join([char for char in text if char.isdigit()])
    #   try:
    #     (float(price))
    #     token['is_price'] = True
    #     token['text'] = price
    #     return token
    #   except:
    #     print(f"Error converting price {text}")

    return token
