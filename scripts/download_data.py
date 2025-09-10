import kaggle

def download_dataset():
    kaggle.api.dataset_download_files(
        'arashnic/book-recommendation-dataset',
        path='data/raw',
        unzip=True,
    )

if __name__ == '__main__':
    download_dataset()