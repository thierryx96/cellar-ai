
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from store.store_loader import WINE_DTYPES, StoreLoader, extract_country, extract_varietal, extract_year

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
        # df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype('Int16')

        # Type and varietal
        df = df.rename(columns={'wine': 'description'})
        df = df.dropna(subset=['description', 'country'])

        df[['type', 'variety']] = pd.DataFrame(df['description'].apply(extract_varietal).tolist(), index=df.index)

        # Country
        df['country'] = df['country'].apply(lambda x: extract_country(x) if pd.notna(x) else None).astype('string')

        # Cleanup / types
        str_cols = [k for k, v in WINE_DTYPES.items() if v == 'string']
        df[str_cols] = df[str_cols].astype('string')
        df = df.drop('wine id', axis=1)

        # Rank
        scaler = MinMaxScaler()
        df['rank'] = scaler.fit_transform(df[['rating']])

        return df