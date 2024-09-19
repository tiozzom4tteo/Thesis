import os

def create_dir(label):
    directory = os.path.join("labels", label)
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_txt_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                content = file.read().split("\t")[1]
                content = content.split("\n")[0]
                create_dir(content)


if __name__ == "__main__":
    directory = "avclass/"
    read_txt_files(directory)