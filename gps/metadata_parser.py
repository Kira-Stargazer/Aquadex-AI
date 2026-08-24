import json


def read_gps_metadata(metadata_file):

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    latitude = data.get("latitude")
    longitude = data.get("longitude")

    return latitude, longitude


if __name__ == "__main__":

    latitude, longitude = read_gps_metadata(
        "metadata.json"
    )

    print("Latitude:", latitude)
    print("Longitude:", longitude)
