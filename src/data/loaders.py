"""Data loading utilities for book recommendation dataset."""

import polars as pl
from pathlib import Path
from typing import Tuple


class BookDataLoader:
    """Loads and validates book recommendation dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
    def load_books(self) -> pl.DataFrame:
        """Load books dataset."""
        books = pl.read_csv(self.data_dir / "Books.csv", 
                           encoding='latin-1',
                           truncate_ragged_lines=True,
                           schema_overrides={"Year-Of-Publication": pl.Utf8})
        return books
    
    def load_ratings(self) -> pl.DataFrame:
        """Load ratings dataset."""
        ratings = pl.read_csv(self.data_dir / "Ratings.csv", 
                             encoding='latin-1')
        return ratings
    
    def load_users(self) -> pl.DataFrame:
        """Load users dataset."""
        users = pl.read_csv(self.data_dir / "Users.csv", 
                           encoding='latin-1')
        return users
    
    def load_all(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load all datasets."""
        return self.load_books(), self.load_ratings(), self.load_users()
    
    def validate_data(self) -> dict:
        """Validate loaded data and return basic stats."""
        books, ratings, users = self.load_all()
        
        stats = {
            'books': {
                'count': books.height,
                'unique_isbns': books['ISBN'].n_unique(),
                'missing_titles': books['Book-Title'].null_count()
            },
            'ratings': {
                'count': ratings.height,
                'unique_users': ratings['User-ID'].n_unique(),
                'unique_books': ratings['ISBN'].n_unique(),
                'rating_range': (ratings['Book-Rating'].min(), ratings['Book-Rating'].max())
            },
            'users': {
                'count': users.height,
                'unique_users': users['User-ID'].n_unique()
            }
        }
        
        return stats