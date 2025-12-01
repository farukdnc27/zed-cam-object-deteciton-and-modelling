#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace sl;
using namespace cv;

class EdgeDetector {
private:
    Camera zed;
    Mat image_left, image_left_ocv;
    Mat edges, contour_image;
    
    // Edge detection parametreleri
    int canny_low_threshold = 50;
    int canny_high_threshold = 150;
    int min_contour_area = 100;

public:
    bool initialize() {
        // ZED kamera başlatma
        InitParameters init_params;
        init_params.camera_resolution = RESOLUTION::HD720;
        init_params.camera_fps = 30;
        init_params.depth_mode = DEPTH_MODE::PERFORMANCE;
        
        ERROR_CODE err = zed.open(init_params);
        if (err != ERROR_CODE::SUCCESS) {
            std::cout << "ZED açılamadı: " << err << std::endl;
            return false;
        }
        
        // Görüntü boyutları
        Resolution image_size = zed.getCameraInformation().camera_resolution;
        image_left.alloc(image_size, MAT_TYPE::U8_C4);
        image_left_ocv = Mat(image_left.getHeight(), image_left.getWidth(), CV_8UC4, image_left.getPtr<sl::uchar1>(MEM::CPU));
        
        return true;
    }
    
    void processFrame() {
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            // Sol görüntüyü al
            zed.retrieveImage(image_left, VIEW::LEFT);
            
            // OpenCV Mat'a dönüştür
            Mat gray_image;
            cvtColor(image_left_ocv, gray_image, COLOR_BGRA2GRAY);
            
            // Gürültüyü azalt
            Mat blurred;
            GaussianBlur(gray_image, blurred, Size(5, 5), 1.5);
            
            // Canny edge detection
            Canny(blurred, edges, canny_low_threshold, canny_high_threshold);
            
            // Morfolojik operasyonlar ile kenarları iyileştir
            Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
            morphologyEx(edges, edges, MORPH_CLOSE, kernel);
            
            // Konturları bul
            std::vector<std::vector<Point>> contours;
            std::vector<Vec4i> hierarchy;
            findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            // Geometrik şekilleri tespit et ve çiz
            detectAndDrawShapes(contours);
        }
    }
    
    void detectAndDrawShapes(const std::vector<std::vector<Point>>& contours) {
        // Sonuç görüntüsünü hazırla
        contour_image = image_left_ocv.clone();
        
        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            
            // Küçük konturları filtrele
            if (area < min_contour_area) continue;
            
            // Kontur yaklaşımı
            std::vector<Point> approx;
            double epsilon = 0.02 * arcLength(contours[i], true);
            approxPolyDP(contours[i], approx, epsilon, true);
            
            // Şekil tipini belirle
            std::string shape_type = classifyShape(approx, area);
            
            // Kontur çiz
            drawContours(contour_image, contours, (int)i, Scalar(0, 255, 0), 2);
            
            // Köşe noktalarını işaretle
            for (const Point& point : approx) {
                circle(contour_image, point, 5, Scalar(0, 0, 255), -1);
            }
            
            // Şekil ismini yaz
            Moments m = moments(contours[i]);
            if (m.m00 != 0) {
                Point center(m.m10/m.m00, m.m01/m.m00);
                putText(contour_image, shape_type, center, 
                       FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);
            }
            
            // Kenar uzunluklarını hesapla ve göster
            showEdgeLengths(approx, contour_image);
        }
    }
    
    std::string classifyShape(const std::vector<Point>& approx, double area) {
        int vertices = approx.size();
        
        if (vertices == 3) {
            return "Ucgen";
        }
        else if (vertices == 4) {
            // Dikdörtgen/kare kontrolü
            Rect rect = boundingRect(approx);
            double aspectRatio = (double)rect.width / rect.height;
            
            if (aspectRatio >= 0.95 && aspectRatio <= 1.05) {
                return "Kare";
            } else {
                return "Dikdortgen";
            }
        }
        else if (vertices == 5) {
            return "Besgen";
        }
        else if (vertices == 6) {
            return "Altigen";
        }
        else if (vertices > 6) {
            // Daire kontrolü
            double perimeter = arcLength(approx, true);
            double circularity = 4 * M_PI * area / (perimeter * perimeter);
            
            if (circularity > 0.75) {
                return "Daire";
            } else {
                return "Cokgen";
            }
        }
        
        return "Bilinmeyen";
    }
    
    void showEdgeLengths(const std::vector<Point>& approx, Mat& image) {
        for (size_t i = 0; i < approx.size(); i++) {
            Point p1 = approx[i];
            Point p2 = approx[(i + 1) % approx.size()];
            
            // Kenar uzunluğu (piksel cinsinden)
            double length = norm(p1 - p2);
            
            // Kenar ortası
            Point mid_point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
            
            // Uzunluğu göster
            std::string length_text = std::to_string((int)length) + "px";
            putText(image, length_text, mid_point, 
                   FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);
            
            // Kenarı vurgula
            line(image, p1, p2, Scalar(255, 0, 255), 2);
        }
    }
    
    void displayResults() {
        // Sonuçları göster
        imshow("ZED - Geometrik Sekil Tespiti", contour_image);
        imshow("Kenarlar", edges);
        
        // Parametre ayarlama için trackbar'lar
        createTrackbar("Canny Low", "ZED - Geometrik Sekil Tespiti", 
                      &canny_low_threshold, 200);
        createTrackbar("Canny High", "ZED - Geometrik Sekil Tespiti", 
                      &canny_high_threshold, 300);
        createTrackbar("Min Alan", "ZED - Geometrik Sekil Tespiti", 
                      &min_contour_area, 1000);
    }
    
    void run() {
        if (!initialize()) {
            return;
        }
        
        std::cout << "ZED Edge Detection başlatıldı..." << std::endl;
        std::cout << "Çıkmak için 'q' tuşuna basın" << std::endl;
        
        while (true) {
            processFrame();
            displayResults();
            
            char key = waitKey(1);
            if (key == 'q' || key == 27) break; // ESC veya 'q' ile çık
        }
        
        zed.close();
        destroyAllWindows();
    }
};

int main() {
    EdgeDetector detector;
    detector.run();
    return 0;
}