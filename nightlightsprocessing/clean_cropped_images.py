import os


def main():
    filepath = "./data/07-cropped-images"
    directories = os.listdir(filepath)
    print("directories", directories)

    for directory in directories:
        path = f"{filepath}/{directory}"

        for root, dirs, files in os.walk(path):
            for name in files:
                if name.endswith((".aux.xml")):
                    os.remove(os.path.join(root, name))


if __name__ == "__main__":
    main()
