
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from store_loader import StoreLoader, extract_varietal, extract_year

class VivinoLoader(StoreLoader):
    """
    Class to load and process Vivino wine data.
    """   

    def __init__(self, path_pattern):
        super().__init__(path_pattern)

    def load(self):
        """
        Load and process Vivino wine data.
        """

        df = super().load()

        # winery          object
        # year            object
        # wine id          int64
        # description     object
        # rating         float64
        # num_review       int64
        # price          float64
        # country         object
        # region          object

        # YEAR
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)

        # Type and varietal
        df = df.rename(columns={'wine': 'description'})
        df = df.dropna(subset=['description'])
        df[['type', 'variety']] = pd.DataFrame(df['description'].apply(extract_varietal).tolist(), index=df.index)


        # Rank
        scaler = MinMaxScaler()
        df['rank'] = scaler.fit_transform(df[['rating']])

        return df