import json
import pandas as pd


def create_json_report(output_file, report):

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )


def create_csv_report(output_file, detections):

    df = pd.DataFrame(detections)

    df.to_csv(
        output_file,
        index=False
    )
