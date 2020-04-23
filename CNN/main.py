from cnn_house_numbers import CNNHouseNumbers
from house_number_detector import HouseNumberDetector

if __name__ == "__main__":
    # specify input and output directories
    # IMG_DIR = ""
    # OUT_DIR = ""
    # cnn = CNNHouseNumbers(img_dir=IMG_DIR, output_dir=OUT_DIR)
    # cnn.preprocess(save_preprocess=True)
    # cnn.load_pre_processed_files()
    # cnn.init_cnn()
    # cnn.train_cnn(save_model=True)
    hnd = HouseNumberDetector()
    hnd.read_imgs(input_dir='input_images')
    hnd.detect_numbers()