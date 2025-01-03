using namespace cv;
using namespace dnn;
using namespace std;

Mat overlayImage(const Mat& background, const Mat& foreground, Point2i location, Size tshirtSize) 
{
    if (background.empty() || foreground.empty()) {
        cerr << "[ERROR] Одно из изображений пустое!" << endl;
        return background.clone();
    }

    if (foreground.channels() != 4) {
        cerr << "[ERROR] Изображение одежды должно иметь 4 канала (RGBA)!" << endl;
        return background.clone();
    }

    Mat output = background.clone();
    Mat resizedTshirt;
    resize(foreground, resizedTshirt, tshirtSize);

    for (int y = 0; y < resizedTshirt.rows; ++y) {
        for (int x = 0; x < resizedTshirt.cols; ++x) {
            int bgY = location.y + y;
            int bgX = location.x + x;

            if (bgY >= 0 && bgY < background.rows && bgX >= 0 && bgX < background.cols) {
                Vec4b fgPixel = resizedTshirt.at<Vec4b>(y, x);
                Vec3b& bgPixel = output.at<Vec3b>(bgY, bgX);

                float alpha = fgPixel[3] / 255.0f;
                for (int c = 0; c < 3; ++c) {
                    bgPixel[c] = saturate_cast<uchar>(alpha * fgPixel[c] + (1.0f - alpha) * bgPixel[c]);
                }
            }
        }
    }
    return output;
}