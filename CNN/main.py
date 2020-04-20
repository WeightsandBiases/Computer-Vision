from cnn_house_numbers import CNNHouseNumbers

if __name__ == "__main__":
    # specify input and output directories
    IMG_DIR = ""
    OUT_DIR = ""
    cnn = CNNHouseNumbers(img_dir_path=IMG_DIR, output_dir_path=OUT_DIR)
    cnn.preprocess()
