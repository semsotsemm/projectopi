#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;

void overlayImage(const  Mat& background, const  Mat& foreground, Mat& output, Point2i location)
{
    background.copyTo(output);

    for (int y = max(location.y, 0); y < background.rows; ++y) {
        int fY = y - location.y;

        if (fY >= foreground.rows) {
            break;
        }

        for (int x = max(location.x, 0); x < background.cols; ++x) {
            int fX = x - location.x;

            if (fX >= foreground.cols) {
                break;
            }

            double opacity = ((double)foreground.at< Vec4b>(fY, fX)[3]) / 255.;

            for (int c = 0; opacity > 0 && c < output.channels(); ++c) {
                output.at< Vec3b>(y, x)[c] =
                    background.at< Vec3b>(y, x)[c] * (1. - opacity) +
                    foreground.at< Vec4b>(fY, fX)[c] * opacity;
            }
        }
    }
}

using namespace dnn;

  vector< Point> detectBodyKeypoints(const  Mat &person) 
  {
      Net net =   readNetFromONNX("pose_estimation.onnx");

    if (net.empty())
    {
          cerr << "Ошибка открытия pose_estimation.onnx" <<   endl;
        return;
    }

     Mat blob =   blobFromImage(person, 1.0 / 255.0,  Size(256, 256),  Scalar(0, 0, 0), true, false);
    net.setInput(blob);

     Mat output = net.forward();

      vector< Point> keypoints;
    for (int i = 0; i < 17; ++i)
    {
        float x = output.at<float>(0, i * 2) * person.cols;
        float y = output.at<float>(0, i * 2 + 1) * person.rows;
        keypoints.emplace_back( Point(static_cast<int>(x), static_cast<int>(y)));
    }
    return keypoints;
}

int main() {
    Mat person = imread("person.jpg");
    Mat tshirt = imread("tshirt.png", IMREAD_UNCHANGED);

    if (person.empty() || tshirt.empty()) {
        cerr << "Ошибка загрузки изображения" << endl;
        return -1;
    }

    /*  vector< Point> keypoints = detectBodyKeypoints(person);
    if (keypoints.empty())
    {
          cerr << "Ошибка поиска ключ.точек" <<   endl;
        return -1;
    }

    int bodyX = keypoints[5].x; // Left shoulder
    int bodyY = keypoints[5].y; // Top of torso
    int bodyWidth = keypoints[6].x - keypoints[5].x; // Distance between shoulders
    int bodyHeight = keypoints[11].y - keypoints[5].y; // From shoulder to hips
    */
    resize(tshirt, tshirt, Size(300, 350));
    Mat result;
    overlayImage(person, tshirt, result, Point(250, 200));

    imwrite("person_with_tshirt.jpg", result);

    cout << "Image saved as person_with_tshirt.jpg" << endl;
    return 0;
}
