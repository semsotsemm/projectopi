#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <OverlayCloth.h>
#include <iostream>
#include <vector>

using namespace cv;
using namespace dnn;
using namespace std;

// --- ������� ����������� �������� ����� ���� ---
vector<Point> detectBodyKeypoints(const Mat& person, const string& modelPath, const string& protoPath) {
    vector<Point> keypoints;

    Net net = readNet(modelPath, protoPath);
    if (net.empty()) {
        cerr << "[ERROR] ������ �������� ������ OpenPose!" << endl;
        return keypoints;
    }

    // �������������� ����������� � ������ ��� ������
    Mat blob;
    blobFromImage(person, blob, 1.0 / 255.0, Size(368, 368), Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    Mat output = net.forward();

    int H = output.size[2]; // ������ �����
    int W = output.size[3]; // ������ �����

    const int NUM_KEYPOINTS = 25;
    for (int i = 0; i < NUM_KEYPOINTS; ++i) {
        Mat heatMap(H, W, CV_32F, output.ptr(0, i));
        Point maxLoc;
        double maxVal;

        minMaxLoc(heatMap, 0, &maxVal, 0, &maxLoc);

        if (maxVal > 0.1) { // ����������� > 0.1
            keypoints.push_back(Point(static_cast<int>(maxLoc.x * person.cols / W),
                static_cast<int>(maxLoc.y * person.rows / H)));
        }
        else {
            keypoints.push_back(Point(-1, -1));
        }
    }

    return keypoints;
}

// --- ������� ���������� ��������� � ������� ����� ---
template <typename T>
constexpr const T& clamp(const T& value, const T& low, const T& high) {
    return (value < low) ? low : (value > high) ? high : value;
}
Point calculateTshirtPosition(vector<Point>& keypoints, Size tshirtSize) {
    if (keypoints[1].x == -1 || keypoints[1].y == -1 ||
        keypoints[2].x == -1 || keypoints[5].x == -1) {
        cerr << "[ERROR] ����� ��� ��� ���� �� ����������!" << endl;
        return Point(0, 0);
    }

    // ��������: ���������� ����� ���� � �� ������ ����� �������
    //int x = keypoints[8].y - tshirtSize.width / 2.75; // ����� �� ����
    int x = ((keypoints[8].x + keypoints[16].x) / 2 - tshirtSize.width / 2); // ����� �� ����
    int y = keypoints[1].y - tshirtSize.height / 10; // ��������: ���� ���
    return Point(x, y);
}

Size calculateTshirtSize(vector<Point>& keypoints, const Mat& tshirt) {
    if (keypoints[2].x == -1 || keypoints[5].x == -1 ||
        keypoints[8].x == -1 || keypoints[8].y == -1) {
        cerr << "[ERROR] ����� ���� ��� ���� �� ����������! ������������ ����������� ������ ������." << endl;
        return Size(tshirt.cols, tshirt.rows);
    }

    // ��������: ����������� ������ �����
    int bodyWidth = abs(keypoints[5].x - keypoints[2].x); // ���������� ����� �������
    int bodyHeight = abs(keypoints[8].y - keypoints[1].y); // ������ �� ��� �� ����

    if (bodyWidth <= 0 || bodyHeight <= 0) {
        cerr << "[ERROR] ������������ ������� ����! ������������ ����������� ������ ������." << endl;
        return Size(tshirt.cols, tshirt.rows);
    }

    // ��������: �������� ������������ ������ � ������
    float scaleFactorWidth = static_cast<float>(bodyWidth) / tshirt.cols * 2.2; // ��������: ���� �� 100%
    float scaleFactorHeight = static_cast<float>(bodyHeight) / tshirt.rows * 1.2; // ������ ��������� �� 20%

    int newWidth = static_cast<int>(tshirt.cols * scaleFactorWidth);
    int newHeight = static_cast<int>(tshirt.rows * scaleFactorHeight);

    // ��������� �������� (������� ��� ����)
    const int minSize = 50;
    const int maxSize = 1000;
    newWidth = clamp(newWidth, minSize, maxSize);
    newHeight = clamp(newHeight, minSize, maxSize);

    return Size(newWidth, newHeight);
}

// --- ������� ���������� ��������� � ������� ������ ---
Point calculatePantsPosition(vector<Point>& keypoints, Size pantsSize) {
    if (keypoints[8].x == -1 || keypoints[8].y == -1 ||
        keypoints[9].x == -1 || keypoints[12].x == -1) {
        cerr << "[ERROR] ����� ���� ��� ����� �� ����������!" << endl;
        return Point(0, 0);
    }

    // ��������� ������: ������� ����� ������� ������ � ����
    int x = ((keypoints[8].x + keypoints[16].x) / 2 - pantsSize.width / 2);
    int y = keypoints[8].y; // ����� ���������� �� ����
    return Point(x, y);
}

Size calculatePantsSize(vector<Point>& keypoints, const Mat& pants) {
    if (keypoints[9].x == -1 || keypoints[12].x == -1 ||
        keypoints[10].y == -1 || keypoints[13].y == -1) {
        cerr << "[ERROR] ����� ����� ��� ������� �� ����������! ������������ ����������� ������ ������." << endl;
        return Size(pants.cols, pants.rows);
    }

    // ������ ������: ���������� ����� �������
    int hipWidth = abs(keypoints[5].x - keypoints[2].x);
    // ������ ������: ���������� �� ���� �� �������
    int pantsHeight = abs(keypoints[10].y - keypoints[24].y);

    if (hipWidth <= 0 || pantsHeight <= 0) {
        cerr << "[ERROR] ������������ ������� ����! ������������ ����������� ������ ������." << endl;
        return Size(pants.cols, pants.rows);
    }

    // ������������ ������ � ������ ��� ��������������� ������
    float scaleFactorWidth = static_cast<float>(hipWidth) / pants.cols * 1.6; // ���������� �� 130%
    float scaleFactorHeight = static_cast<float>(pantsHeight) / pants.rows * 2; // ���������� �� 130%

    int newWidth = static_cast<int>(pants.cols * scaleFactorWidth);
    int newHeight = static_cast<int>(pants.rows * scaleFactorHeight);

    // ��������� ��������
    const int minSize = 50;
    const int maxSize = 1000;
    newWidth = clamp(newWidth, minSize, maxSize);
    newHeight = clamp(newHeight, minSize, maxSize);

    return Size(newWidth, newHeight);
}

// ---- ����������� �������� ����� ----
void drawKeypoints(Mat& image, const vector<Point>& keypoints) {
    Scalar usedKeypointColor(0, 0, 0);    // ������ ��� �����, ������� ������������
    Scalar otherKeypointColor(255, 255, 255); // ����� ��� ��������� �����
    int radius = 5;
    int thickness = -1;

    for (int i = 0; i < keypoints.size(); ++i) {
        if (keypoints[i].x != -1 && keypoints[i].y != -1) {
            Scalar color = (i == 1 || i == 2 || i == 5 || i == 8 || i == 24) ? usedKeypointColor : otherKeypointColor;
            circle(image, keypoints[i], radius, color, thickness);

            // ����������� ������ �������� �����
            putText(image, to_string(i), keypoints[i] + Point(5, -5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        }
    }
}

// --- ������� ���������� ��������� � ������� ����� ---
Point calculateHatPosition(vector<Point>& keypoints, Size hatSize) {
    if (keypoints[0].x == -1 || keypoints[0].y == -1) { // ����� ������
        cerr << "[ERROR] ����� ������ �� ����������!" << endl;
        return Point(0, 0);
    }

    int x = keypoints[16].x - hatSize.width / 2; // ����� �� ������
    int y = keypoints[16].y - hatSize.height;   // ��� �������
    return Point(x, y);
}

Size calculateHatSize(vector<Point>& keypoints, const Mat& hat) {
    if (keypoints[0].x == -1 || keypoints[0].y == -1 || keypoints[1].x == -1 || keypoints[1].y == -1) {
        cerr << "[ERROR] ����� ������ �� ����������! ������������ ����������� ������ �����." << endl;
        return Size(hat.cols, hat.rows);
    }

    int headWidth = abs(keypoints[16].x - keypoints[17].x) * 2.5; // ��������� ������ ������
    float scaleFactor = static_cast<float>(headWidth) / hat.cols;

    int newWidth = static_cast<int>(hat.cols * scaleFactor);
    int newHeight = static_cast<int>(hat.rows * scaleFactor);

    const int minSize = 50;
    const int maxSize = 500;
    newWidth = clamp(newWidth, minSize, maxSize);
    newHeight = clamp(newHeight, minSize, maxSize);

    return Size(newWidth, newHeight);
}

// --- ������� ���������� ��������� � ������� ����� ---
Point calculateGlassesPosition(vector<Point>& keypoints, Size glassesSize) {
    if (keypoints[1].x == -1 || keypoints[1].y == -1 || keypoints[2].x == -1 || keypoints[5].x == -1) {
        cerr << "[ERROR] ����� ���� ��� ������ �� ����������!" << endl;
        return Point(0, 0);
    }

    int x = (keypoints[0].x + keypoints[18].x) / 2 - glassesSize.width / 2; // ����� ����� �������
    int y = keypoints[18].y - glassesSize.height / 3.5; // ���� ���� ������� ����� ������
    return Point(x, y);
}

Size calculateGlassesSize(vector<Point>& keypoints, const Mat& glasses) {
    if (keypoints[1].x == -1 || keypoints[2].x == -1 || keypoints[5].x == -1) {
        cerr << "[ERROR] ����� ���� �� ����������! ������������ ����������� ������ �����." << endl;
        return Size(glasses.cols, glasses.rows);
    }

    int eyeDistance = abs(keypoints[18].x - keypoints[0].x); // ���������� ����� �������
    float scaleFactor = static_cast<float>(eyeDistance) / glasses.cols * 2.2;

    int newWidth = static_cast<int>(glasses.cols * scaleFactor);
    int newHeight = static_cast<int>(glasses.rows * scaleFactor);

    const int minSize = 30;
    const int maxSize = 300;
    newWidth = clamp(newWidth, minSize, maxSize);
    newHeight = clamp(newHeight, minSize, maxSize);

    return Size(newWidth, newHeight);
}

// --- ������� ������� (�����������) ---
int main() {
    setlocale(LC_ALL, "Russian");

    string personPath = "D:/Proj/Project/x64/Debug/person.jpg";
    string tshirtPath = "D:/Proj/Project/x64/Debug/tshirt2.png";
    string pantsPath = "D:/Proj/Project/x64/Debug/pants.png";
    string hatPath = "D:/Proj/Project/x64/Debug/hat2.png";
    string glassesPath = "D:/Proj/Project/x64/Debug/glasses.png";
    string modelPath = "D:/Proj/Project/x64/Debug/pose_iter_584000.caffemodel";
    string protoPath = "D:/Proj/Project/openpose/models/pose/body_25/pose_deploy.prototxt";

    Mat person = imread(personPath);
    if (person.empty()) {
        cerr << "[ERROR] �� ������� ��������� ����������� ��������!" << endl;
        return -1;
    }

    Mat tshirt = imread(tshirtPath, IMREAD_UNCHANGED);
    Mat pants = imread(pantsPath, IMREAD_UNCHANGED);
    Mat hat = imread(hatPath, IMREAD_UNCHANGED);
    Mat glasses = imread(glassesPath, IMREAD_UNCHANGED);

    vector<Point> keypoints = detectBodyKeypoints(person, modelPath, protoPath);
    if (keypoints.empty()) {
        cerr << "[ERROR] �� ������� ���������� �������� �����!" << endl;
        return -1;
    }

    drawKeypoints(person, keypoints);

    // �����
    Size tshirtSize = calculateTshirtSize(keypoints, tshirt);
    Point tshirtLocation = calculateTshirtPosition(keypoints, tshirtSize);
    Mat output = overlayImage(person, tshirt, tshirtLocation, tshirtSize);

    // �����
    Size pantsSize = calculatePantsSize(keypoints, pants);
    Point pantsLocation = calculatePantsPosition(keypoints, pantsSize);
    output = overlayImage(output, pants, pantsLocation, pantsSize);

    // �����
    Size hatSize = calculateHatSize(keypoints, hat);
    Point hatLocation = calculateHatPosition(keypoints, hatSize);
    output = overlayImage(output, hat, hatLocation, hatSize);

    // ����
    Size glassesSize = calculateGlassesSize(keypoints, glasses);
    Point glassesLocation = calculateGlassesPosition(keypoints, glassesSize);
    output = overlayImage(output, glasses, glassesLocation, glassesSize);

    imwrite("result_with_all_items.jpg", output);

    imshow("Result", output);
    waitKey(0);

    return 0;
}
