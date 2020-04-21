from cnn_house_numbers import CNNHouseNumbers

if __name__ == "__main__":
    # specify input and output directories
    IMG_DIR = ""
    OUT_DIR = ""
    cnn = CNNHouseNumbers(img_dir=IMG_DIR, output_dir=OUT_DIR)
    cnn.preprocess(save_preprocess=True)
    cnn.load_pre_processed_files()
    cnn.init_cnn()
    cnn.run_cnn()
