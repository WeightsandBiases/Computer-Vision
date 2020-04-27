from cnn_house_numbers import CNNHouseNumbers
from house_number_detector import HouseNumberDetector

if __name__ == "__main__":
    # specify input and output directories
    cnn = CNNHouseNumbers(training_img_dir="training_images", tf_model_dir="tf_model")
    cnn.preprocess(save_preprocess=True)
    cnn.load_pre_processed_files()
    cnn.init_cnn()
    cnn.train_cnn(save_model=True)
    hnd = HouseNumberDetector(
        input_dir="input_images", tf_model_dir="tf_model", output_dir="output_images"
    )
    hnd.read_imgs()
    hnd.detect_numbers()
