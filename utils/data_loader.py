import pandas as pd

def load_dataset(file):
    import pandas as pd

    if file is None:
        raise ValueError("No file uploaded")

    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")

    return df
