"""Data processing utilities for book recommendation dataset."""

import polars as pl
import re
from typing import Tuple, Dict, Any


class BookDataProcessor:
    """Processes and cleans book recommendation data."""
    
    def __init__(self, min_ratings_per_user: int = 5, min_ratings_per_book: int = 5):
        self.min_ratings_per_user = min_ratings_per_user
        self.min_ratings_per_book = min_ratings_per_book
    
    def clean_books(self, books: pl.DataFrame) -> pl.DataFrame:
        """Clean books dataset."""
        return books.with_columns([
            # Clean year column - extract 4-digit years, convert invalid to null
            pl.col("Year-Of-Publication")
            .str.extract(r"(\d{4})")
            .cast(pl.Int32, strict=False)
            .alias("year"),
            
            # Clean book titles - strip whitespace, handle nulls
            pl.col("Book-Title").str.strip_chars().alias("title"),
            
            # Clean author names
            pl.col("Book-Author").str.strip_chars().alias("author"),
            
            # Keep ISBN as string for joining
            pl.col("ISBN").alias("isbn"),
            
            # Clean publisher
            pl.col("Publisher").str.strip_chars().alias("publisher")
        ]).select(["isbn", "title", "author", "year", "publisher"])
    
    def clean_ratings(self, ratings: pl.DataFrame) -> pl.DataFrame:
        """Standardize columns of ratings dataset."""
        return ratings.with_columns([
            pl.col("User-ID").alias("user_id"),
            pl.col("ISBN").alias("isbn"),
            pl.col("Book-Rating").alias("rating")
        ]).select(["user_id", "isbn", "rating"])
    
    def clean_users(self, users: pl.DataFrame) -> pl.DataFrame:
        """Clean users dataset."""
        return users.with_columns([
            pl.col("User-ID").alias("user_id"),
            pl.col("Location").str.strip_chars().alias("location"),
            pl.col("Age").cast(pl.Int32, strict=False).alias("age")
        ]).select(["user_id", "location", "age"])
    
    def filter_sparse_data(self, ratings: pl.DataFrame) -> pl.DataFrame:
        """Remove records (could be users or books) with too few ratings."""
        # Count ratings per user and book
        user_counts = ratings.group_by("user_id").len().rename({"len": "user_rating_count"})
        book_counts = ratings.group_by("isbn").len().rename({"len": "book_rating_count"})
        
        # Filter iteratively
        prev_shape = (0, 0)
        while ratings.shape != prev_shape:
            prev_shape = ratings.shape
            
            # Filter users with enough ratings
            ratings = ratings.join(user_counts, on="user_id").filter(
                pl.col("user_rating_count") >= self.min_ratings_per_user
            ).drop("user_rating_count")
            
            # Filter books with enough ratings
            ratings = ratings.join(book_counts, on="isbn").filter(
                pl.col("book_rating_count") >= self.min_ratings_per_book
            ).drop("book_rating_count")
            
            # Recalculate counts
            user_counts = ratings.group_by("user_id").len().rename({"len": "user_rating_count"})
            book_counts = ratings.group_by("isbn").len().rename({"len": "book_rating_count"})
        
        return ratings
    
    def create_interaction_matrix(self, ratings: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """Create user-item interaction matrix with mappings."""
        # Create user and item ID mappings
        unique_users = ratings["user_id"].unique().sort()
        unique_books = ratings["isbn"].unique().sort()
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users.to_list())}
        book_to_idx = {book: idx for idx, book in enumerate(unique_books.to_list())}
        
        # Add mapped indices
        interaction_matrix = ratings.with_columns([
            pl.col("user_id").map_elements(lambda x: user_to_idx[x], return_dtype=pl.Int32).alias("user_idx"),
            pl.col("isbn").map_elements(lambda x: book_to_idx[x], return_dtype=pl.Int32).alias("book_idx")
        ])
        
        mappings = {
            "user_to_idx": user_to_idx,
            "book_to_idx": book_to_idx,
            "idx_to_user": {v: k for k, v in user_to_idx.items()},
            "idx_to_book": {v: k for k, v in book_to_idx.items()},
            "n_users": len(unique_users),
            "n_books": len(unique_books)
        }
        
        return interaction_matrix, mappings
    
    def create_llm_context_features(self, books: pl.DataFrame, users: pl.DataFrame, 
                                   ratings: pl.DataFrame) -> pl.DataFrame:
        """Create features for LLM context generation."""
        # User reading profile
        user_profiles = ratings.group_by("user_id").agg([
            pl.col("rating").mean().alias("avg_rating"),
            pl.col("rating").count().alias("total_ratings")
        ])
        
        # Book statistics
        book_stats = ratings.group_by("isbn").agg([
            pl.col("rating").mean().alias("avg_rating"),
            pl.col("rating").count().alias("total_ratings"),
            pl.col("rating").std().alias("rating_std")
        ])
        
        # Combine with metadata
        enhanced_books = books.join(book_stats, on="isbn")
        enhanced_users = users.join(user_profiles, on="user_id")
        
        return enhanced_books, enhanced_users
    
    def process_all(self, books: pl.DataFrame, ratings: pl.DataFrame, 
                   users: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
        """Process all datasets."""
        # Clean datasets
        clean_books = self.clean_books(books)
        clean_ratings = self.clean_ratings(ratings)
        clean_users = self.clean_users(users)
        
        # Filter sparse data
        filtered_ratings = self.filter_sparse_data(clean_ratings)
        
        # Keep only books and users that appear in filtered ratings
        valid_books = clean_books.filter(
            pl.col("isbn").is_in(filtered_ratings["isbn"])
        )
        valid_users = clean_users.filter(
            pl.col("user_id").is_in(filtered_ratings["user_id"])
        )
        
        # Create interaction matrix and mappings
        interaction_matrix, mappings = self.create_interaction_matrix(filtered_ratings)
        
        # Create LLM context features
        enhanced_books, enhanced_users = self.create_llm_context_features(
            valid_books, valid_users, filtered_ratings
        )
        
        return enhanced_books, filtered_ratings, enhanced_users, mappings