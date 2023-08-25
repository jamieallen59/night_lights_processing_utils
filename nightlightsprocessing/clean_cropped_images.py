import os


def main():
    filepath = "./data/07-cropped-images"
    directories = os.listdir(filepath)
    print("directories", directories)

    for directory in directories:
        if not directory.startswith("."):
            path = f"{filepath}/{directory}"
            sub_directories = os.listdir(path)

            for location_sub_directory in sub_directories:
                if not location_sub_directory.startswith("."):
                    for root, dirs, files in os.walk(f"{path}/{location_sub_directory}"):
                        for name in files:
                            if name.endswith((".aux.xml")):
                                os.remove(os.path.join(root, name))


if __name__ == "__main__":
    main()
