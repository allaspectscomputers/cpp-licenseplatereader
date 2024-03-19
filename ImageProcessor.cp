class ImageProcessor {
public:
    static void preprocessImage(const cv::Mat &img, cv::Mat &processedImg) {
        cv::Mat gray, blurred, edges;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        cv::adaptiveThreshold(blurred, edges, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
        processedImg = edges;
    }
};
