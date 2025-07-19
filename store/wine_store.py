from fuzzywuzzy import fuzz
import pandas as pd
from store.store_loader import WINE_DTYPES 

class WineStore:
    """
    Class to load and process Vivino wine data.
    """   

    def __init__(self, path):
        self.path = path
        self.db = None

    def load(self):
        self.db = pd.read_csv(self.path, dtype=WINE_DTYPES)

    def retrieve_wine(self, menu_entry):
        """
        Find the best matching wine from database for a menu entry.
        
        Args:
            menu_entry: dict with keys like 'text', 'year', 'region', 'country', 'variety'
        
        Returns:
            tuple: (best_match_row, best_score)
        """
        if self.db is None:
            raise ValueError("Database not loaded. Call load() first.")
            
        best_score = 0
        best_match = None
        df = self.db.copy()
        if 'country' in menu_entry and 'type' in menu_entry and (menu_entry['type'] == 'red' or menu_entry['type'] == 'white'):
          country = menu_entry['country']
          type = menu_entry['type']

          df = df[(df["country"] == country) & (df["type"] == type)]
          print(f"filter by country {country} and {type}, size {len(df)}/{len(self.db)}")

        else:
          return None, 0
        
        for i, wine in df.iterrows():
            score = self._calculate_match_score(menu_entry, wine)
            
            if score > best_score:
                best_score = score
                best_match = wine
        
        return best_match, best_score

    def _calculate_match_score(self, menu_entry, wine_row):
        """
        Calculate weighted similarity score between menu entry and wine.
        
        Returns:
            float: Score between 0 and 1
        """
        score = 0
        
        # Text similarity (heaviest weight)
        if 'description' in menu_entry and 'description' in wine_row and pd.notna(wine_row['description']):
            text_sim = fuzz.ratio(str(menu_entry['description']), str(wine_row['description'])) / 100
            score += text_sim * 0.4

        # Text similarity (heaviest weight)
        if 'description' in menu_entry and 'winery' in wine_row and pd.notna(wine_row['winery']):
            text_sim = fuzz.ratio(str(menu_entry['description']), str(wine_row['winery'])) / 100
            score += text_sim * 0.3
        
        # Year match
        if 'year' in menu_entry and 'year' in wine_row and pd.notna(wine_row['year']):
            if menu_entry['year'] == wine_row['year']:
                score += 0.2
            elif abs(menu_entry['year'] - wine_row['year']) <= 1:  # ±1 year tolerance
                score += 0.1
        
        # Region match
        if 'region' in menu_entry and 'region' in wine_row and pd.notna(wine_row['region']):
            region_sim = fuzz.ratio(str(menu_entry['region']), str(wine_row['region'])) / 100
            score += region_sim * 0.40
        
        # Country match
        if 'country' in menu_entry and 'country' in wine_row and pd.notna(wine_row['country']):
            country_sim = fuzz.ratio(str(menu_entry['country']), str(wine_row['country'])) / 100
            score += country_sim * 0.20
        
        # Variety match
        if 'variety' in menu_entry and 'variety' in wine_row and pd.notna(wine_row['variety']):
            variety_sim = fuzz.ratio(str(menu_entry['variety']), str(wine_row['variety'])) / 100
            score += variety_sim * 0.1
        
        # Variety match
        if 'type' in menu_entry and 'type' in wine_row and pd.notna(wine_row['type']):
            type_sim = fuzz.ratio(str(menu_entry['type']), str(wine_row['type'])) / 100
            score += type_sim * 0.1



        return score