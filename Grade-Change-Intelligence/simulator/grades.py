
import pandas as pd


def load_grades(path="data/grades.csv"):
    """
    Loads all grade recipes.
    """

    return pd.read_csv(path)


def get_grade(df, grade_code):
    """
    Returns one recipe as a dictionary.
    """

    row = df[df["grade_code"] == grade_code]

    if row.empty:
        raise ValueError(f"Grade '{grade_code}' not found.")

    return row.iloc[0].to_dict()