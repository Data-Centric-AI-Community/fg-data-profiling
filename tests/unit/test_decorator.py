import pandas as pd

from data_profiling import ProfileReport


def test_decorator(tmp_path):
    people_example = tmp_path / "people_example.csv"
    pd.DataFrame(
        {
            "First Name": ["Bob", "Alice", "Malcolm", "Felix", "Alex", "Tod", "Derek"],
            "Last Name": [
                "Smith",
                "Williams",
                "Jone",
                "Brown",
                "Smith",
                "Jones",
                "Smith",
            ],
            "Country": [
                "United States",
                "Canada",
                "England",
                "USA",
                "Canada",
                "USA",
                "England",
            ],
            "age": [24, 23, 22, 26, 22, 23, 24],
        }
    ).to_csv(people_example, index=False)
    df = pd.read_csv(people_example)
    report = ProfileReport(
        df,
        title="Coursera Test Report",
        samples={"head": 20},
        missing_diagrams={"heatmap": False},
    )
    assert "Coursera Test Report" in report.to_html(), "Title is not found"
