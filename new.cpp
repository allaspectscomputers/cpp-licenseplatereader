#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

class ImageProcessor {
public:
    static void preprocessImage(const cv::Mat &img, cv::Mat &processedImg) {
        if (img.empty()) {
            std::cerr << "Image is empty, cannot preprocess." << std::endl;
            return;
        }
        // Convert to grayscale
        cv::cvtColor(img, processedImg, cv::COLOR_BGR2GRAY);
        // Reduce noise with a Gaussian filter
        cv::GaussianBlur(processedImg, processedImg, cv::Size(5, 5), 0);
        // Apply adaptive thresholding to binarize the image
        cv::adaptiveThreshold(processedImg, processedImg, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
        // Edge detection to enhance character edges
        cv::Canny(processedImg, processedImg, 100, 200);
        // Morphological operations to help isolate characters
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(processedImg, processedImg, cv::MORPH_CLOSE, element);
        cv::morphologyEx(processedImg, processedImg, cv::MORPH_OPEN, element);
    }
};

class FeatureExtractor {
public:
    static cv::Mat extractFeatures(const cv::Mat &img) {
        cv::HOGDescriptor hog;
        std::vector<float> descriptors;
        hog.compute(img, descriptors);
        if (descriptors.empty())
            return cv::Mat::zeros(1, 3780, CV_32F);
        return cv::Mat(descriptors).reshape(1, 1);
    }
};

class CharacterSegmenter {
public:
    static void segmentCharacters(const cv::Mat &plate, std::vector<cv::Mat> &charImgs) {
        cv::Mat plateCopy;
        cv::cvtColor(plate, plateCopy, cv::COLOR_BGR2GRAY);
        cv::threshold(plateCopy, plateCopy, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(plateCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::vector<cv::Rect> boundRects;
        for (auto &contour : contours) {
            boundRects.push_back(cv::boundingRect(contour));
        }
        sort(boundRects.begin(), boundRects.end(), [](const cv::Rect &a, const cv::Rect &b) {
            return a.x < b.x;
        });
        for (auto &rect : boundRects) {
            charImgs.push_back(plate(rect));
        }
    }
};

class SVMClassifier {
    cv::Ptr<cv::ml::SVM> svm;

public:
    SVMClassifier() {
        svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::LINEAR);
        svm->setC(1);
        svm->setGamma(0.5);
        svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10000, 1e-6));
    }

    void train(const cv::Mat &trainData, const cv::Mat &labels) {
        svm->trainAuto(trainData, cv::ml::ROW_SAMPLE, labels);
    }

    float predict(const cv::Mat &features) {
        return svm->predict(features);
    }

    void saveModel(const std::string &path) {
        svm->save(path);
    }

    void loadModel(const std::string &path) {
        svm = cv::ml::SVM::load(path);
    }
};

class PlateRecognizer {
    ImageProcessor imgProcessor;
    FeatureExtractor featureExtractor;
    CharacterSegmenter charSegmenter;
    SVMClassifier svmClassifier;

public:
    PlateRecognizer(const std::string &modelPath) {
        svmClassifier.loadModel(modelPath);
    }

    void recognizePlate(const cv::Mat &img) {
        cv::Mat processedImg;
        ImageProcessor::preprocessImage(img, processedImg);
        std::vector<cv::Mat> charImgs;
        CharacterSegmenter::segmentCharacters(processedImg, charImgs);
        std::cout << "Recognized characters: ";
        for (auto &charImg : charImgs) {
            cv::Mat features = FeatureExtractor::extractFeatures(charImg);
            float prediction = svmClassifier.predict(features);
            std::cout << static_cast<char>(prediction);
        }
        std::cout << std::endl;
    }
};

int main() {
    PlateRecognizer recognizer("model.yml");
    cv::Mat img = cv::imread("path_to_license_plate_image.jpg");
    if (img.empty()) {
        std::cout << "Could not read the image." << std::endl;
        return -1;
    }
    recognizer.recognizePlate(img);
    return 0;
}
