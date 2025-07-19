
import pandas as pd
from store.store_loader import WINE_DTYPES, StoreLoader, extract_country, extract_varietal, extract_year

class WinemagLoader(StoreLoader):
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

        # country                   object
        # description               object
        # designation               object
        # points                     int64
        # price                    float64
        # province                  object
        # region_1                  object
        # region_2                  object
        # variety                   object
        # winery                    object
        # taster_name               object
        # taster_twitter_handle     object
        # title                     object

        df = df.rename(columns={'description': 'note', 'designation' : 'description', 'province' : 'region'})
        str_cols = [k for k, v in WINE_DTYPES.items() if v == 'string']
        df[str_cols] = df[str_cols].astype('string')

        # Country
        df = df.dropna(subset=['country'])
        df['country'] = df['country'].apply(lambda x: extract_country(x) if pd.notna(x) else None).astype('string')


        # df.dtypes.head(20)
        # df[['type', 'variety']] = pd.DataFrame(df['description'].apply(extract_varietal).tolist(), index=df.index)

        def combine_description(row):
          if pd.notna(row['title']) and pd.notna(row['description']):
            if row['description'] in row['title']:
              return row['title']
            elif row['title'] not in row['description']: 
              return row['title'] + ' ' + row['description']
            else:
              return row['description']
          
          elif pd.notna(row['title']):
              return row['title']
          
          elif pd.notna(row['description']):
              return row['description']

        df['description'] = df.apply(combine_description, axis=1).astype('string')

        df['year'] = df['description'].apply(lambda x: extract_year(x) if pd.notna(x) else None).astype('Int16')

        df[['type', 'variety_2']] = pd.DataFrame(df['description'].apply(lambda x: extract_varietal(x) if pd.notna(x) else (None, None)).tolist(), index=df.index)
        df[['type_2', 'variety_3']] = pd.DataFrame(df['variety'].apply(lambda x: extract_varietal(x) if pd.notna(x) else (None, None)).tolist(), index=df.index)

        df['variety'] = df['variety_2'].combine_first(df['variety_3']).combine_first(df['variety']).astype('string')
        df['type'] = df['type'].combine_first(df['type_2']).astype('string')

        df = df.drop(['type_2', 'variety_2', 'variety_3', 'taster_name', 'taster_twitter_handle'], axis=1)

        # scaler = MinMaxScaler(feature_range=(0.5, ))
        df['rank'] = df['points'] / 100.0


        return df